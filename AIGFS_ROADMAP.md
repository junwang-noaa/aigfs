<p align="center">
<h2 align="center">AIGFS Roadmap</h2>

## Introduction
AI/ML technology is changing the landscape of weather prediction. NOAA is actively developing a new generation of data-driven models. Recently, the AI Global Forecast System (AIGFS) version 1.0 has been implemented with significantly improved synoptic forecast skills and resource efficiency. It complements the existing NOAA GFS physics-based NWP model forecast.

This document serves as a roadmap for the next 2-5 years of AIGFS development. It includes current status (v1.0), next upgrade, and new features afterwards. The initial plan is to have an annual release with changes well tested and then deployed to operational implementation. The work will be coordinated through a project board.

This roadmap will be updated annually due to the fast evolvement of AI/ML weather model development.

## Goals
The primary objective of AIGFS development is to advance medium-range global forecasting through AI/ML innovation. This includes expansion of forecast products and significantly sharpening predictive capabilities for hurricanes, winter storms, and other high-impact extreme weather events.

## Agile Development

Incremental Progress: All work is to be done incrementally, never standing in the way of any other work or release.

Experimental Forecasts: Experimental near-real-time forecasts from development versions will be made available through the EAGLE project on NODD. Corresponding evaluations against operational GFS and AIGFS will also be provided.

Documentation & Testing: No code modifications will take place without first having complete documentation and testing of the affected code.

Responsibility: Effective code documentation is the responsibility of every developer to enhance readability, maintainability, and collaboration. Developers are encouraged to write tests and documentation to ensure code correctness and robustness.


CI/CD: Operational standards are integrated through a CI/CD pipeline during the model development process to streamline fast deployment

## Current Status (AIGFS v1.0)

AIGFS was implemented into opration in December 17, 2025. 
Model: Leverages Google DeepMind GraphCast, fine-tuned with GDAS data as inputs and ERA5/HRES/GDAS as training targets (Tabas et al., 2025).

Output: Provides 6-hourly, 16-day forecasts at 0.25-degree resolution on surface and 13 pressure levels.

Variables: Forecast fields include 6 atmospheric model state variables and 6 surface fields.

Performance: Outperforms operational GFS with significantly improved synoptic-scale patterns, 2-m temperature, and 10-m winds. Hurricane track forecasts are significantly better than those from GFS. Major weaknesses are the degraded hyrricane intensity forecasts and blurring issue in the long lead forecast time.

## Near term plans
Following AIGFS v1 implementation, continuous development will be carried out to add new features in the EAGLE SOLO system and then transition to operation. Immediate goals include improving hurricane intensity forecasts, producing more products, and sharpening the long lead time forecasts.

### Repository Integration
- Collaborate with EPIC and OAR labs to integrate operational code into a common repository.

- Build the Anemoi framework, including the capability to run operational AI models.

### New Features

- Dataset Updates
Update training dataset with the latest 2024-2025 GFSv16 data and GFSv17 retrospective analysis using UFS2ARCO.
Collaborate with partners to create a common cloud-based data source for sharing target and evaluation data within the community.

- Model Updates
    * Fine-tune GraphCast with new datasets.
    * Update loss function (variable scaling, AMSE, etc.).
    * Implement physically constrained training to remove negative values and ensure physically meaningful forecasts.
    * Add more prognostic and diagnostic fields, including variables to drive WRF models.

- Post-Processing (UPP)
    * Use UPP to output additional diagnostic products.

- Evaluation
    * Implement real-time verification into the operational EMC Verification System (EVS).
    * Add new features, such as scorecards, to METplus and EVS.
    * Develop near-real-time evaluation systems and webpages with skill maps.
    * Seek feedback from forecasters on model performance during model development.

- Implement a DevOps approach between development and operational evaluation systems.

- Infrastructure
    * Build and update the Anemoi common training framework.
    * Merge in-house AIGFS/AIGEFS training script transitions.


## Future Development

New techniques will be applied to the AIGFS model. It includes:

### Integrating of physics-informed constraints, optimizing loss functions, addressing stochastic uncertainty, updating ML architecture, and improving scalability
- Implement global mass and energy conservation constraints

### Develop AI fully coupled earth modeling system to include other model components for wave, land, ocean, and seaice etc.
-  Add marine components for a coupled system (e.g., wave height, period, and direction)

### Explore increase model spatial(horizontal and vertical) and temporal resolution. The technical resolution limits for AIGFS in particular with respect to training the model will be investigated and an optimal vertical and temporal resolutions will be decided.
-  Increase vertical resolution from 13 to 37 levels to provide better vertical profiles.
-  Increase horizontal resolution from 13 to 37 levels to provide better vertical profiles.

- Infrastructure
    * Extend GPU parallelization capabilities during the training process.

## Releases
All new features will be integrated incrementally. Beta versions will be released as individual or combined features demonstrate improved performance, followed by an annual formal release for operational implementation

## Stakeholders & Responsibilities
The development team is charged with executing and delivering the goals outlined above.

Key Entities: 
MDC: 
OAR: 
EPIC: 
NCO: 

## Schedule
Milestones for the near term plans

--------------------------------------------------------------------------------------------
|   Quarter   |                    Goals                              |     Notes          | 
--------------------------------------------------------------------------------------------
| 2025Q2      | Improve hurricane intensity by a) using AMSE loss     |                    |
|             | function 2) updating variable scaling 3) adding       |                    |
|             | recent data sets in training data                     |                    |
--------------------------------------------------------------------------------------------
| 2025Q3      | 1. Generate retrospective run and evaluate results    |                    |
|             | 2. Transition to Anemoi framework, confirm model      |                    |
|             |    inference and training in Anemoi framework         |                    |
--------------------------------------------------------------------------------------------
|             | Implement the develop version  into operation         |                    |                    |
--------------------------------------------------------------------------------------------
|             |                                                       |                    |
--------------------------------------------------------------------------------------------

## Repositories

- AIGFS Repository: https://github.com/noaa-emc/aigfs 
- MLGlobal Repository: https://github.com/NOAA-EMC/MLGlobal

Note: Anemoi repository link to be added
