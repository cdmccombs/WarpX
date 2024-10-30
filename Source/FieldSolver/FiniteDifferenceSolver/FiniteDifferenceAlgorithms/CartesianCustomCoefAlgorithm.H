/* Copyright 2020 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_CUSTOM_COEF_H_
#define WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_CUSTOM_COEF_H_

#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <array>


/**
 * This struct contains only static functions to initialize the stencil coefficients
 * and to compute finite-difference derivatives for a stencil with user-defined coefficents algorithm.
 */
struct CartesianCustomCoefAlgorithm {

    static void InitializeStencilCoefficients (
        std::array<amrex::Real, 3>& cell_size,
        amrex::Vector<amrex::Real>& stencil_coefs_x,
        amrex::Vector<amrex::Real>& stencil_coefs_y,
        amrex::Vector<amrex::Real>& stencil_coefs_z ) {

        using namespace amrex;

        // Compute Cole-Karkkainen-Cowan coefficients according
        // to Cowan - PRST-AB 16, 041303 (2013)
        Real const inv_dx = 1._rt/cell_size[0];
        Real const inv_dy = 1._rt/cell_size[1];
        Real const inv_dz = 1._rt/cell_size[2];

        stencil_coefs_x.resize(2);
        stencil_coefs_x[0] = 1._rt*inv_dx;
        stencil_coefs_x[1] = 1._rt*inv_dx;
        stencil_coefs_y.resize(2);
        stencil_coefs_y[0] = 1._rt*inv_dy;
        stencil_coefs_y[1] = 1._rt*inv_dy;
        stencil_coefs_z.resize(2);
        stencil_coefs_z[0] = 1._rt*inv_dz;
        stencil_coefs_z[1] = 1._rt*inv_dz;

    }

    /**
     * Compute the maximum timestep, for which the scheme remains stable
     * (Courant-Friedrichs-Levy limit) */
    static amrex::Real ComputeMaxDt ( amrex::Real const * const dx ) {
#if (defined WARPX_DIM_1D_Z)
            amrex::Real const delta_t = dx[0]/PhysConst::c;
#elif (defined WARPX_DIM_XZ)
            // - In Cartesian 2D geometry: determined by the minimum cell size in all direction
            amrex::Real const delta_t = std::min( dx[0], dx[1] )/PhysConst::c;
#else
            // - In Cartesian 3D geometry: determined by the minimum cell size in all direction
            amrex::Real const delta_t = std::min( dx[0], std::min( dx[1], dx[2] ) ) / PhysConst::c;
#endif
        return delta_t;
    }

    /**
     * \brief Returns maximum number of guard cells required by the field-solve
     */
    static amrex::IntVect GetMaxGuardCell () {
        // TODO
        return amrex::IntVect{AMREX_D_DECL(1,1,1)};
    }

    /**
     * Perform derivative along x on a cell-centered grid, from a nodal field `F` */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real UpwardDx (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_x, int const n_coefs_x,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
#if defined WARPX_DIM_3D
        amrex::Real derivative = 0;
        for (int l=0; l<n_coefs_x/2; l++) {
            derivative += coefs_x[l] * (F(i+1+l,j,k,ncomp) - F(i-l,j,k,ncomp));
        }
        return derivative;
#endif
    }

    /**
     * Perform derivative along x on a nodal grid, from a cell-centered field `F` */
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real DownwardDx (
        T_Field const& F,
        amrex::Real const * const coefs_x, int const n_coefs_x,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
#if defined WARPX_DIM_3D
        amrex::Real derivative = 0;
        for (int l=n_coefs_x/2; l<n_coefs_x; l++) {
            derivative += coefs_x[l] * (F(i+l,j,k,ncomp) - F(i-l-1,j,k,ncomp));
        }
        return derivative;
#endif
    }

    /**
     * Perform derivative along y on a cell-centered grid, from a nodal field `F` */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real UpwardDy (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_y, int const n_coefs_y,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
#if defined WARPX_DIM_3D
        Real const alphay = coefs_y[1];
        Real const betayz = coefs_y[2];
        Real const betayx = coefs_y[3];
        Real const gammay = coefs_y[4];
        amrex::ignore_unused(n_coefs_y);
        return alphay * (F(i  ,j+1,k  ,ncomp) - F(i  ,j  ,k  ,ncomp))
             + betayx * (F(i+1,j+1,k  ,ncomp) - F(i+1,j  ,k  ,ncomp)
                      +  F(i-1,j+1,k  ,ncomp) - F(i-1,j  ,k  ,ncomp))
             + betayz * (F(i  ,j+1,k+1,ncomp) - F(i  ,j  ,k+1,ncomp)
                      +  F(i  ,j+1,k-1,ncomp) - F(i  ,j  ,k-1,ncomp))
             + gammay * (F(i+1,j+1,k+1,ncomp) - F(i+1,j  ,k+1,ncomp)
                      +  F(i-1,j+1,k+1,ncomp) - F(i-1,j  ,k+1,ncomp)
                      +  F(i+1,j+1,k-1,ncomp) - F(i+1,j  ,k-1,ncomp)
                       +  F(i-1,j+1,k-1,ncomp) - F(i-1,j  ,k-1,ncomp));
#elif (defined WARPX_DIM_XZ || WARPX_DIM_1D_Z)
        amrex::ignore_unused(F, coefs_y, n_coefs_y,
            i, j, k, ncomp);
        return 0._rt; // 1D and 2D Cartesian: derivative along y is 0
#else
        amrex::ignore_unused(F, coefs_y, n_coefs_y,
            i, j, k, ncomp);
#endif
    }

    /**
     * Perform derivative along y on a nodal grid, from a cell-centered field `F` */
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real DownwardDy (
        T_Field const& F,
        amrex::Real const * const coefs_y, int const n_coefs_y,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;
#if defined WARPX_DIM_3D
        amrex::Real const inv_dy = coefs_y[0];
        amrex::ignore_unused(n_coefs_y);
        return inv_dy*( F(i,j,k,ncomp) - F(i,j-1,k,ncomp) );
#elif (defined WARPX_DIM_XZ || WARPX_DIM_1D_Z)
        amrex::ignore_unused(F, coefs_y, n_coefs_y,
            i, j, k, ncomp);
        return 0._rt; // 2D Cartesian: derivative along y is 0
#else
        amrex::ignore_unused(F, coefs_y, n_coefs_y,
            i, j, k, ncomp);
#endif
    }

    /**
     * Perform derivative along z on a cell-centered grid, from a nodal field `F` */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real UpwardDz (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_z, int const n_coefs_z,
        int const i, int const j, int const k, int const ncomp=0 ) {

        using namespace amrex;

        amrex::ignore_unused(n_coefs_z);

        Real const alphaz = coefs_z[1];
#if (defined WARPX_DIM_3D || WARPX_DIM_XZ)
        Real const betazx = coefs_z[2];
#endif
#if defined WARPX_DIM_3D
        Real const betazy = coefs_z[3];
        Real const gammaz = coefs_z[4];
#endif
#if defined WARPX_DIM_3D
        return alphaz * (F(i  ,j  ,k+1,ncomp) - F(i  ,j  ,k  ,ncomp))
             + betazx * (F(i+1,j  ,k+1,ncomp) - F(i+1,j  ,k  ,ncomp)
                      +  F(i-1,j  ,k+1,ncomp) - F(i-1,j  ,k  ,ncomp))
             + betazy * (F(i  ,j+1,k+1,ncomp) - F(i  ,j+1,k  ,ncomp)
                      +  F(i  ,j-1,k+1,ncomp) - F(i  ,j-1,k  ,ncomp))
             + gammaz * (F(i+1,j+1,k+1,ncomp) - F(i+1,j+1,k  ,ncomp)
                      +  F(i-1,j+1,k+1,ncomp) - F(i-1,j+1,k  ,ncomp)
                      +  F(i+1,j-1,k+1,ncomp) - F(i+1,j-1,k  ,ncomp)
                      +  F(i-1,j-1,k+1,ncomp) - F(i-1,j-1,k  ,ncomp));
#elif (defined WARPX_DIM_XZ)
        return alphaz * (F(i  ,j+1,k  ,ncomp) - F(i  ,j  ,k  ,ncomp))
             + betazx * (F(i+1,j+1,k  ,ncomp) - F(i+1,j  ,k  ,ncomp)
                      +  F(i-1,j+1,k  ,ncomp) - F(i-1,j  ,k  ,ncomp));
#elif (defined WARPX_DIM_1D_Z)
        return alphaz * (F(i+1  ,j,k  ,ncomp) - F(i  ,j  ,k  ,ncomp));
#endif
    }

    /**
     * Perform derivative along z on a nodal grid, from a cell-centered field `F` */
    template< typename T_Field>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real DownwardDz (
        T_Field const& F,
        amrex::Real const * const coefs_z, int const /*n_coefs_z*/,
        int const i, int const j, int const k, int const ncomp=0) {

        amrex::Real const inv_dz = coefs_z[0];
#if defined WARPX_DIM_3D
        return inv_dz*( F(i,j,k,ncomp) - F(i,j,k-1,ncomp) );
#elif (defined WARPX_DIM_XZ)
        return inv_dz*( F(i,j,k,ncomp) - F(i,j-1,k,ncomp) );
#elif (defined WARPX_DIM_1D_Z)
        return inv_dz*( F(i,j,k,ncomp) - F(i-1,j,k,ncomp) );
#endif
    }

};

#endif // WARPX_FINITE_DIFFERENCE_ALGORITHM_CARTESIAN_CUSTOM_COEF_H_
