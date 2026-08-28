/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_TFSF_SOURCE_H_
#define WARPX_TFSF_SOURCE_H_

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Geometry.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

#include <array>
#include <memory>

/**
 * \brief Total-field/scattered-field (TFSF, Huygens-surface) injection of an
 *        analytically prescribed incident wave through a closed (or partially
 *        closed) surface placed a few cells inside the domain boundary.
 *
 *        The cells on the interior side of the surface hold the total field;
 *        the cells between the surface and the domain boundary (typically
 *        backed by PML) hold the scattered field only.  The split is enforced
 *        by correcting the Yee finite-difference stencils that straddle the
 *        surface, using the incident field evaluated from user-provided
 *        parsers.  Nothing is overwritten: unlike a Dirichlet
 *        (pec_insulator) drive, outgoing/scattered light crosses the surface
 *        unimpeded and is absorbed by the PML behind it.
 *
 *        Only the explicit staggered (Yee) Cartesian solver is supported
 *        (1D, 2D-XZ, 3D), on a single level.  The macroscopic solver is
 *        supported provided the TFSF surface itself sits in vacuum.
 *
 *        Input parameters (all under the `tfsf.` prefix):
 *          tfsf.faces          = x_lo x_hi z_lo ...   (subset of the domain faces;
 *                                                      presence enables the source)
 *          tfsf.offset_cells   = 8                    (cells between domain edge and surface)
 *          tfsf.Ex_inc_function(x,y,z,t) = ...        (incident-field components;
 *          ... Ey, Ez, Bx, By, Bz                      unspecified components are zero)
 */
class TFSFSource
{
public:

    TFSFSource ();

    /** Whether any TFSF face was requested in the input file */
    [[nodiscard]] bool isEnabled () const { return m_enabled; }

    /**
     * \brief Add the incident-field correction to the E components whose
     *        curl-B stencil straddles the TFSF surface.  Must be called after
     *        the E-field update of the solver, with the same dt.
     *
     * \param[in,out] Efield   the E field being corrected
     * \param[in]     geom     geometry of the level
     * \param[in]     dt       time step of the E update
     * \param[in]     t_Binc   time at which the B field read by the E update
     *                         is centered (start_time + dt/2 for leapfrog)
     */
    void ApplyToE (ablastr::fields::VectorField const& Efield,
                   amrex::Geometry const& geom,
                   amrex::Real dt, amrex::Real t_Binc) const;

    /**
     * \brief Add the incident-field correction to the B components whose
     *        curl-E stencil straddles the TFSF surface.  Must be called after
     *        the B-field update of the solver, with the same dt.
     *
     * \param[in,out] Bfield   the B field being corrected
     * \param[in]     geom     geometry of the level
     * \param[in]     dt       time step of the B update
     * \param[in]     t_Einc   time at which the E field read by the B update
     *                         is centered (start_time for the first half push,
     *                         start_time + dt for the second half push)
     */
    void ApplyToB (ablastr::fields::VectorField const& Bfield,
                   amrex::Geometry const& geom,
                   amrex::Real dt, amrex::Real t_Einc) const;

    /**
     * \brief Shared implementation: loop over the enabled faces and apply the
     *        one-cell-deep stencil corrections for either the E or the B push.
     *        Public because nvcc requires the enclosing function of an
     *        extended device lambda to be publicly accessible.
     *
     * \param[in,out] field    E field (E_like) or B field (!E_like)
     * \param[in]     geom     geometry of the level
     * \param[in]     dt       time step of the update
     * \param[in]     t_inc    time at which the partner incident field is evaluated
     * \param[in]     E_like   true when correcting the E push (partner is B_inc)
     */
    void ApplyCorrections (ablastr::fields::VectorField const& field,
                           amrex::Geometry const& geom,
                           amrex::Real dt, amrex::Real t_inc,
                           bool E_like) const;

private:

    bool m_enabled = false;
    int m_offset_cells = 8;

    //! Face-enabled flags, indexed [physical dimension 0..2][side 0=lo,1=hi]
    std::array<std::array<bool,2>,3> m_face_enabled {{ {{false,false}}, {{false,false}}, {{false,false}} }};

    //! Incident-field parsers: Ex, Ey, Ez, Bx, By, Bz (nullptr when not given)
    std::array<std::unique_ptr<amrex::Parser>,6> m_parsers;
    std::array<amrex::ParserExecutor<4>,6> m_executors;
    std::array<bool,6> m_has_component {{false, false, false, false, false, false}};
};

#endif // WARPX_TFSF_SOURCE_H_
