/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "SplitAndScatterFunc.H"

SplitAndScatterFunc::SplitAndScatterFunc (const std::string& collision_name,
                                          MultiParticleContainer const * const mypc):
    m_collision_type{BinaryCollisionUtils::get_collision_type(collision_name, mypc)}
{
    if (m_collision_type == CollisionType::DSMC)
    {
        const amrex::ParmParse pp_collision_name(collision_name);

        // Build the ScatteringProcess objects using the same shared helper as DSMCFunc, so
        // that the process ordering matches the indices encoded in the per-pair mask. The
        // scatter kernel uses these to look up, for each colliding pair, the process type,
        // scattering angle model and energy penalty.
        m_scattering_processes = BinaryCollisionUtils::parse_scattering_processes(collision_name);
#ifdef AMREX_USE_GPU
        amrex::Gpu::HostVector<ScatteringProcess::Executor> h_scattering_processes_exe;
        for (auto const& p : m_scattering_processes) {
            h_scattering_processes_exe.push_back(p.executor());
        }
        m_scattering_processes_exe.resize(h_scattering_processes_exe.size());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_scattering_processes_exe.begin(),
                              h_scattering_processes_exe.end(), m_scattering_processes_exe.begin());
        amrex::Gpu::streamSynchronize();
#else
        for (auto const& p : m_scattering_processes) {
            m_scattering_processes_exe.push_back(p.executor());
        }
#endif

        // Check if the scattering processes include reactions that produce macroparticles in new species
        // (i.e. not in the incident species list), i.e. if it contains ionization, charge exchange or two-product reaction
        amrex::Vector<std::string> scattering_processes;
        pp_collision_name.queryarr("scattering_processes", scattering_processes);
        const bool reaction_produces_new_species = std::any_of(scattering_processes.begin(), scattering_processes.end(), [](const std::string& process) {
            return process == "ionization" || process == "charge_exchange" || process == "two_product_reaction" || process == "dissociation";
        });

        if (reaction_produces_new_species) {

            // Check that product species have been specified
            amrex::Vector<std::string> product_species;
            pp_collision_name.getarr("product_species", product_species);
            // TODO: check number of product species

            // For ionization:
            if (std::find(scattering_processes.begin(), scattering_processes.end(), "ionization") != scattering_processes.end()) {
                m_num_product_species = 4;
                m_num_products_host.push_back(1); // slot 0: first reactant (incident/non-target) species; 1 scattered particle per ionization event
                m_num_products_host.push_back(0); // slot 1: other reactant (target) species; consumed by reaction, no new reaction-produced particle
                m_num_products_host.push_back(1); // slot 2: first true product species (e.g. ejected electron)
                m_num_products_host.push_back(1); // slot 3: second true product species (e.g. resulting ion)
            }

            // For charge exchange or two-product reaction:
            if (std::find(scattering_processes.begin(), scattering_processes.end(), "charge_exchange") != scattering_processes.end() ||
                std::find(scattering_processes.begin(), scattering_processes.end(), "two_product_reaction") != scattering_processes.end()) {
                m_num_product_species = 4;
                m_num_products_host.push_back(0); // slot 0: first reactant species; consumed by reaction, no new reaction-produced particles
                m_num_products_host.push_back(0); // slot 1: other reactant species; consumed by reaction, no new reaction-produced particles
                m_num_products_host.push_back(1); // slot 2: first true product species
                m_num_products_host.push_back(1); // slot 3: second true product species
            }

            // For dissociation (e.g. e- + H2 -> e- + H + H):
            if (std::find(scattering_processes.begin(), scattering_processes.end(), "dissociation") != scattering_processes.end()) {
                m_num_product_species = 4;
                m_num_products_host.push_back(1); // slot 0: incident (non-target) species, e.g. the electron; survives the collision
                m_num_products_host.push_back(0); // slot 1: target species (the molecule); consumed by the reaction
                m_num_products_host.push_back(1); // slot 2: first neutral fragment
                m_num_products_host.push_back(1); // slot 3: second neutral fragment

                // Energy (eV) the incident particle loses to break up the molecule (the impact
                // threshold, e.g. ~8.5 eV for the H2 b3Sigma_u+ channel). The cross-section must
                // be 0 at this energy (enforced in ScatteringProcess::init).
                pp_collision_name.get("dissociation_energy", m_reaction_energy);
                // Optional Franck-Condon kinetic-energy release (eV) shared back-to-back by the two
                // fragments (e.g. ~4 eV = threshold - bond energy for H2). Defaults to 0.
                pp_collision_name.query("dissociation_fragment_energy", m_fragment_energy);
            }

        } else {
            m_num_product_species = 2;
            m_num_products_host.push_back(0);
            m_num_products_host.push_back(0);
        }
    }
    else
    {
        WARPX_ABORT_WITH_MESSAGE("Unknown collision type in SplitAndScatterFunc");
    }
}
