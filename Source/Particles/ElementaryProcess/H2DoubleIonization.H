/* Copyright 2026
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 *
 * H2 double-ionization split process:
 *     H2+  ->  H+ + H+ + e-
 * Triggered by the level-1 (W2) entry of the species' custom rate
 * coefficients (see IonizationFilterFunc with min_level = 1). The source
 * molecule is removed; the two protons are emitted back-to-back in the
 * molecular CM frame with kinetic energy release (KER) sampled from a
 * CTMC-derived inverse-CDF table; the electron inherits the molecule's
 * velocity (standard ADK-PIC birth-at-rest in the parent frame).
 *
 * Used with filterCopyTransformParticles<2>: each selected H2+ produces
 * TWO copies in the ion destination (both kept, given +-dv) and TWO copies
 * in the electron destination (second one invalidated).
 */
#ifndef WARPX_H2_DOUBLE_IONIZATION_H_
#define WARPX_H2_DOUBLE_IONIZATION_H_

#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>
#include <AMReX_Random.H>

#include <cmath>

struct H2DISplitTransformFunc
{
    const amrex::Real* AMREX_RESTRICT m_ker_cdf; //!< KER [J] at uniform quantiles
    int m_n_cdf;
    amrex::ParticleReal m_ion_mass; //!< product ion (proton) mass [kg]

    H2DISplitTransformFunc (const amrex::Real* a_ker_cdf, int a_n_cdf,
                            amrex::ParticleReal a_ion_mass) noexcept
        : m_ker_cdf{a_ker_cdf}, m_n_cdf{a_n_cdf}, m_ion_mass{a_ion_mass} {}

    template <typename DstData, typename SrcData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (DstData& dst_ion, DstData& dst_ele, SrcData& src,
                     int i_src, int i_ion, int i_ele,
                     amrex::RandomEngine const& engine) const noexcept
    {
        using namespace amrex::literals;

        // --- KER from inverse CDF (linear interpolation between quantiles)
        const amrex::Real u = amrex::Random(engine) * (m_n_cdf - 1);
        const int iq = static_cast<int>(u) >= m_n_cdf - 1 ? m_n_cdf - 2
                                                          : static_cast<int>(u);
        const amrex::Real frac = u - iq;
        const amrex::Real ker = m_ker_cdf[iq] * (1._rt - frac)
                              + m_ker_cdf[iq + 1] * frac;

        // Each proton carries KER/2 in the CM frame: dv = sqrt(KER/m)
        const auto dv = static_cast<amrex::ParticleReal>(
            std::sqrt(ker / m_ion_mass));

        // --- random isotropic emission direction
        const amrex::Real cth = 1._rt - 2._rt * amrex::Random(engine);
        const amrex::Real sth = std::sqrt(std::max(0._rt, 1._rt - cth * cth));
        const amrex::Real phi = 2._rt * MathConst::pi * amrex::Random(engine);
        const auto nx = static_cast<amrex::ParticleReal>(sth * std::cos(phi));
        const auto ny = static_cast<amrex::ParticleReal>(sth * std::sin(phi));
        const auto nz = static_cast<amrex::ParticleReal>(cth);

        // --- protons: back-to-back kick on top of the copied CM velocity
        dst_ion.m_rdata[PIdx::ux][i_ion]     += dv * nx;
        dst_ion.m_rdata[PIdx::uy][i_ion]     += dv * ny;
        dst_ion.m_rdata[PIdx::uz][i_ion]     += dv * nz;
        dst_ion.m_rdata[PIdx::ux][i_ion + 1] -= dv * nx;
        dst_ion.m_rdata[PIdx::uy][i_ion + 1] -= dv * ny;
        dst_ion.m_rdata[PIdx::uz][i_ion + 1] -= dv * nz;

        // --- electron: first copy kept as-is (inherits molecule velocity);
        //     second copy is an artifact of the N=2 copy -> invalidate it.
        dst_ele.m_rdata[PIdx::w][i_ele + 1] = 0._rt;
        dst_ele.m_idcpu[i_ele + 1] = amrex::ParticleIdCpus::Invalid;

        // --- remove the source molecule (weight zeroed so any deposition
        //     before the next Redistribute sees nothing)
        src.m_rdata[PIdx::w][i_src] = 0._rt;
        src.m_idcpu[i_src] = amrex::ParticleIdCpus::Invalid;
    }
};

#endif // WARPX_H2_DOUBLE_IONIZATION_H_
