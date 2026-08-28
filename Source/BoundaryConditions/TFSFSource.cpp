/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "BoundaryConditions/TFSFSource.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <string>
#include <vector>

namespace
{
    // Map from physical dimension (0=x, 1=y, 2=z) to the array dimension of
    // the AMReX arrays, or -1 when the physical dimension does not exist.
    constexpr int PhysToArrayDim (int pd)
    {
#if defined(WARPX_DIM_3D)
        return pd;
#elif defined(WARPX_DIM_XZ)
        return (pd == 0) ? 0 : ((pd == 2) ? 1 : -1);
#elif defined(WARPX_DIM_1D_Z)
        return (pd == 2) ? 0 : -1;
#else
        amrex::ignore_unused(pd);
        return -1;
#endif
    }

    // Read one optional incident-field component parser
    bool ReadComponentParser (amrex::ParmParse const & pp,
                              std::unique_ptr<amrex::Parser> & parser,
                              std::string const & input_name)
    {
        std::string str = "";
        bool const specified = utils::parser::Query_parserString(pp, input_name, str);
        if (specified) {
            parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(str, {"x", "y", "z", "t"}));
        }
        return specified;
    }
}

TFSFSource::TFSFSource ()
{
    amrex::ParmParse const pp_tfsf("tfsf");

    std::vector<std::string> face_names;
    pp_tfsf.queryarr("faces", face_names);
    m_enabled = !face_names.empty();
    if (!m_enabled) { return; }

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    WARPX_ABORT_WITH_MESSAGE("tfsf.faces: the TFSF source is not implemented for cylindrical/spherical geometry");
#endif

    for (auto const & name : face_names) {
        int pd = -1;
        int side = -1;
        if      (name == "x_lo") { pd = 0; side = 0; }
        else if (name == "x_hi") { pd = 0; side = 1; }
        else if (name == "y_lo") { pd = 1; side = 0; }
        else if (name == "y_hi") { pd = 1; side = 1; }
        else if (name == "z_lo") { pd = 2; side = 0; }
        else if (name == "z_hi") { pd = 2; side = 1; }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(pd >= 0,
            "tfsf.faces: unrecognized face name " + name);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(::PhysToArrayDim(pd) >= 0,
            "tfsf.faces: face " + name + " does not exist for this dimensionality");
        m_face_enabled[pd][side] = true;
    }

    utils::parser::queryWithParser(pp_tfsf, "offset_cells", m_offset_cells);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_offset_cells >= 1,
        "tfsf.offset_cells must be >= 1 so that the TFSF surface lies inside the domain");

    m_has_component[0] = ::ReadComponentParser(pp_tfsf, m_parsers[0], "Ex_inc_function(x,y,z,t)");
    m_has_component[1] = ::ReadComponentParser(pp_tfsf, m_parsers[1], "Ey_inc_function(x,y,z,t)");
    m_has_component[2] = ::ReadComponentParser(pp_tfsf, m_parsers[2], "Ez_inc_function(x,y,z,t)");
    m_has_component[3] = ::ReadComponentParser(pp_tfsf, m_parsers[3], "Bx_inc_function(x,y,z,t)");
    m_has_component[4] = ::ReadComponentParser(pp_tfsf, m_parsers[4], "By_inc_function(x,y,z,t)");
    m_has_component[5] = ::ReadComponentParser(pp_tfsf, m_parsers[5], "Bz_inc_function(x,y,z,t)");

    for (int n = 0; n < 6; ++n) {
        if (m_has_component[n]) {
            m_executors[n] = m_parsers[n]->compile<4>();
        }
    }
}

void
TFSFSource::ApplyToE (ablastr::fields::VectorField const& Efield,
                      amrex::Geometry const& geom,
                      amrex::Real dt, amrex::Real t_Binc) const
{
    ApplyCorrections(Efield, geom, dt, t_Binc, true);
}

void
TFSFSource::ApplyToB (ablastr::fields::VectorField const& Bfield,
                      amrex::Geometry const& geom,
                      amrex::Real dt, amrex::Real t_Einc) const
{
    ApplyCorrections(Bfield, geom, dt, t_Einc, false);
}

void
TFSFSource::ApplyCorrections (ablastr::fields::VectorField const& field,
                              amrex::Geometry const& geom,
                              amrex::Real dt, amrex::Real t_inc,
                              bool E_like) const
{
    using namespace amrex::literals;

    amrex::Box const& domain = geom.Domain();
    auto const plo = geom.ProbLoArray();
    auto const dx = geom.CellSizeArray();

    // Total-field region in node-index space, per physical dimension.
    // In directions without a TFSF face the layers span the valid domain.
    int tf_lo_node[3];
    int tf_hi_node[3];
    for (int pd = 0; pd < 3; ++pd) {
        int const ad = ::PhysToArrayDim(pd);
        if (ad < 0) { tf_lo_node[pd] = 0; tf_hi_node[pd] = 0; continue; }
        tf_lo_node[pd] = domain.smallEnd(ad) + (m_face_enabled[pd][0] ? m_offset_cells : 0);
        tf_hi_node[pd] = domain.bigEnd(ad) + 1 - (m_face_enabled[pd][1] ? m_offset_cells : 0);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(tf_lo_node[pd] < tf_hi_node[pd],
            "TFSF surface offsets leave no total-field region");
    }

    for (int pd = 0; pd < 3; ++pd) {
        int const ad = ::PhysToArrayDim(pd);
        if (ad < 0) { continue; }
        for (int side = 0; side < 2; ++side) {
            if (!m_face_enabled[pd][side]) { continue; }

            int const s = (side == 0) ? tf_lo_node[pd] : tf_hi_node[pd];
            amrex::Real const sgn = (side == 0) ? 1._rt : -1._rt;
            amrex::Real const node_coord = plo[ad]
                + (s - domain.smallEnd(ad)) * dx[ad];
            int const e1 = (pd + 1) % 3;
            int const e2 = (pd + 2) % 3;

            // The two corrections on this face:
            //   E push:  E_e1 += sgn*c^2*dt/dx * B_e2,inc   and   E_e2 -= sgn*c^2*dt/dx * B_e1,inc
            //   B push:  B_e2 += sgn*dt/dx * E_e1,inc       and   B_e1 -= sgn*dt/dx * E_e2,inc
            struct Correction { int target; int partner; amrex::Real sign; };
            Correction corrections[2];
            amrex::Real coef;
            if (E_like) {
                corrections[0] = {e1, 3 + e2,  sgn};
                corrections[1] = {e2, 3 + e1, -sgn};
                coef = PhysConst::c * PhysConst::c * dt / dx[ad];
            } else {
                corrections[0] = {e2, e1,  sgn};
                corrections[1] = {e1, e2, -sgn};
                coef = dt / dx[ad];
            }

            for (auto const & corr : corrections) {
                if (!m_has_component[corr.partner]) { continue; }

                amrex::MultiFab * mf = field[corr.target];
                amrex::IndexType const itype = mf->ixType();

                // Location of the layer in the face-normal direction, and
                // the face-normal coordinate at which the partner incident
                // component (one half-cell across the surface for the E push,
                // on the surface for the B push) is evaluated
                int i_fix;
                amrex::Real partner_coord;
                if (E_like) {
                    // Tangential E on the surface is nodal in the normal direction
                    AMREX_ALWAYS_ASSERT(itype.nodeCentered(ad));
                    i_fix = s;
                    partner_coord = node_coord - sgn * 0.5_rt * dx[ad];
                } else {
                    // Tangential B just outside the surface is staggered in the normal direction
                    AMREX_ALWAYS_ASSERT(itype.cellCentered(ad));
                    i_fix = (side == 0) ? s - 1 : s;
                    partner_coord = node_coord;
                }

                // Build the layer box: fixed in the normal direction, spanning
                // the total-field region (per this component's staggering) in
                // the transverse directions
                amrex::IntVect layer_lo, layer_hi;
                for (int pd2 = 0; pd2 < 3; ++pd2) {
                    int const ad2 = ::PhysToArrayDim(pd2);
                    if (ad2 < 0) { continue; }
                    if (ad2 == ad) {
                        layer_lo[ad2] = i_fix;
                        layer_hi[ad2] = i_fix;
                    } else {
                        layer_lo[ad2] = tf_lo_node[pd2];
                        layer_hi[ad2] = tf_hi_node[pd2] - (itype.cellCentered(ad2) ? 1 : 0);
                    }
                }
                amrex::Box const layer(layer_lo, layer_hi, itype);

                auto const & exec = m_executors[corr.partner];
                amrex::Real const signed_coef = corr.sign * coef;
                amrex::IntVect const dlo = domain.smallEnd();
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> shift;
                for (int ad2 = 0; ad2 < AMREX_SPACEDIM; ++ad2) {
                    shift[ad2] = itype.cellCentered(ad2) ? 0.5_rt : 0._rt;
                }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (amrex::MFIter mfi(*mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    amrex::Box const bx = mfi.tilebox() & layer;
                    if (!bx.ok()) { continue; }
                    amrex::Array4<amrex::Real> const & fab = mf->array(mfi);
                    amrex::ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        {
                            amrex::ignore_unused(j, k);
                            amrex::Real x = 0._rt;
                            amrex::Real y = 0._rt;
                            amrex::Real z = 0._rt;
#if defined(WARPX_DIM_3D)
                            x = plo[0] + (i - dlo[0] + shift[0]) * dx[0];
                            y = plo[1] + (j - dlo[1] + shift[1]) * dx[1];
                            z = plo[2] + (k - dlo[2] + shift[2]) * dx[2];
#elif defined(WARPX_DIM_XZ)
                            x = plo[0] + (i - dlo[0] + shift[0]) * dx[0];
                            z = plo[1] + (j - dlo[1] + shift[1]) * dx[1];
#elif defined(WARPX_DIM_1D_Z)
                            z = plo[0] + (i - dlo[0] + shift[0]) * dx[0];
#endif
                            // The partner incident component lives across (E push)
                            // or on (B push) the surface in the normal direction
                            if      (pd == 0) { x = partner_coord; }
                            else if (pd == 1) { y = partner_coord; }
                            else              { z = partner_coord; }

                            fab(i, j, k) += signed_coef * exec(x, y, z, t_inc);
                        });
                }
            }
        }
    }
}
