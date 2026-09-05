.. raw:: html

   <div style="text-align: center; margin: 40px 0;">
     <img src="_static/corr_favicon.png" alt="CORR-Vars Logo" style="width: 120px; height: 120px; margin-bottom: 20px;">
     <h1 style="font-size: 3em; margin: 0; color: #2c3e50;">CORR-Variables</h1>
     <p style="font-size: 1.3em; color: #7f8c8d; margin: 10px 0 30px 0;">
       Streamlining clinical research with real-world data
     </p>
   </div>

.. admonition:: 🚀 **What is CORR-Vars?**
   :class: note

   CORR-Variables is a **Python package** for building clinical study cohorts from ICU and hospital data sources, developed by the **Charité Outcomes Research Repository (CORR)** team.

   It functions as a high-level connector on top of a data source, preprocessing raw clinical data into **clinically meaningful, quality-checked variables** whose definitions are served and versioned by the CORR Concepts API, to streamline research with real-world data.

.. grid:: 1 1 3 3

   .. grid-item-card:: 🏥 **Clinical Focus**
      :text-align: center
      :class-header: bg-primary text-white
      
      Pre-defined clinical variables validated by medical experts

   .. grid-item-card:: ⚡ **High Performance**
      :text-align: center
      :class-header: bg-success text-white
      
      Built on Polars for fast processing of large datasets

   .. grid-item-card:: 🔗 **Easy Integration**
      :text-align: center
      :class-header: bg-info text-white
      
      Simple API that works with existing analysis workflows


Quick Start
-----------

.. tab-set::

   .. tab-item:: 📦 **Install**

      .. code-block:: bash

         uv add git+https://github.com/CUB-CORR/corr-vars.git

      Requires Python ≥ 3.10.

   .. tab-item:: 🔐 **Concepts API access**

      Variable definitions are served by the CORR Concepts API. A cohort needs a
      project and a key:

      .. code-block:: bash

         export CORR_CONCEPTS_API_KEY="corr_..."

      The project is passed per cohort: ``Cohort(project="my_project")``.


Your First CORR Cohort
---------------------------

Get started in under 5 minutes:

.. code-block:: python

   # Import the main class
   from corr_vars import Cohort
   
   # Create your first cohort
   cohort = Cohort(
       obs_level="icu_stay",
       sources={"reprodicu": {"path": "/path/to/reprodICU_files"}},
       project="my_project",
   )
   
   # Add clinical variables
   cohort.add_variable("age_on_admission")
   cohort.add_variable("blood_sodium")
   
   # View your data
   print(f"Cohort: {len(cohort.obs)} patients")
   print(cohort.obs.head())

.. admonition:: 🎯 **Next Steps**
   :class: tip

   * **New to CORR-Vars?** → Start with our :doc:`tutorials`
   * **Need a specific variable?** → Browse the `Concept Browser <https://concepts.example.edu>`_
   * **Want to contribute?** → Read our :doc:`contributing_variables` guide

Documentation Structure
-----------------------

.. grid:: 1 1 2 2

   .. grid-item-card:: 📚 **Learning Resources**
      :link: tutorials
      :link-type: doc
      :class-header: bg-primary text-white
      
      * **Getting Started Tutorial** - Your first analysis in 30 minutes
      * **Custom Variables Guide** - Create your own clinical variables  
      * **Contributing Guide** - Add variables to the community catalog
      * **Troubleshooting** - Solutions for common issues

   .. grid-item-card:: 🔧 **API Documentation**
      :link: api/cohort
      :link-type: doc
      :class-header: bg-info text-white
      
      * **Cohort Class** - Main interface for building cohorts
      * **Variable Types** - Native, derived, and aggregation variables
      * **Data Sources** - ReprodICU, the local source skeleton, and the source plugin system
      * **Legacy Interface** - Pandas compatibility layer

.. dropdown:: 📖 **Complete Table of Contents**
   :color: light
   :icon: book

   .. toctree::
      :maxdepth: 2
      :caption: Learning Resources
      
      tutorials
      custom_variables
      contributing_variables
      troubleshooting

   .. toctree::
      :maxdepth: 2
      :caption: API Reference
      
      api/cohort
      api/core
      api/sources
      api/utils

Core Architecture
-----------------

.. grid:: 1 1 2 2

   .. grid-item-card:: 🏥 **Observation Levels**
      :class-header: bg-primary text-white
      
      **Choose your analysis unit:**
      
      * **Patient** - One row per unique patient (``patient_id``)
      * **Hospital Stay** - Complete hospitalization periods (``case_id``)
      * **ICU Stay** - Individual intensive care episodes (``icu_stay_id``)
      * **Procedure** - Specific surgical/medical procedures (``procedure_id``)
      
      .. image:: _static/cv_obs_levels.png
         :width: 100%

   .. grid-item-card:: 📊 **Variable Types**
      :class-header: bg-success text-white
      
      **Rich clinical data hierarchy:**
      
      * **Native** - Direct database extractions
      * **Derived** - Computed from existing variables
      * **Static** - Single values per observation
      * **Dynamic** - Time-series measurements
      
      .. image:: _static/cv_var_hierarchy.png
         :width: 100%

.. admonition:: 🔍 **Explore Available Variables**
   :class: tip

   Browse our **300+** pre-defined clinical variables in the interactive `Concept Browser <https://concepts.example.edu>`_

Real-World Example
------------------

Here's how researchers use CORR-Vars for clinical studies:

.. code-block:: python

   # Build an ICU sepsis cohort
   cohort = Cohort(
       obs_level="icu_stay",
       sources={"reprodicu": {"path": "/path/to/reprodICU_files"}},
       project="my_project",
   )
   
   
   # Add a static variable
   cohort.add_variable("sofa_score_imputed") 
   
   # Add time-series biomarkers
   cohort.add_variable("blood_lactate")
   cohort.add_variable("blood_creatinine")
   
   # Apply inclusion criteria
   cohort.include_list([
       {"variable": "age_on_admission", "operation": ">= 18", "label": "Adults"},
       {"variable": "sofa_score_imputed", "operation": ">= 2", "label": "Organ dysfunction"}
   ])
   
   # Generate publication-ready summary
   table1 = cohort.tableone(groupby="inhospital_death")
   print(f"Study cohort: {len(cohort.obs)} patients")

.. admonition:: 📈 **Publication Ready**
   :class: note

   CORR-Vars concepts are quality-checked by attending physicians at Charité Berlin before being used for:
   
   * Critical care outcomes research
   * Machine learning model development  
   * Health services research
   * Quality improvement studies


Community & Support
-------------------

.. grid:: 1 1 3 3

   .. grid-item-card:: 🐛 **Found a Bug?**
      :link: https://github.com/cub-corr/corr-vars/issues
      :link-type: url
      :class-header: bg-warning text-white
      
      Report issues or request features on GitHub

   .. grid-item-card:: 💬 **Need Help?**
      :class-header: bg-info text-white
      
      Check our :doc:`troubleshooting` guide or contact the team

   .. grid-item-card:: 🤝 **Want to Contribute?**
      :link: contributing_variables
      :link-type: doc
      :class-header: bg-success text-white
      
      Add new variables to help the research community

---

.. raw:: html

   <div style="background: #004d9b; color: white; padding: 30px; border-radius: 15px; margin: 40px 0; text-align: center;">
     <h3 style="margin: 0 0 15px 0; color: white;">🏥 Developed at Charité Berlin</h3>
     <p style="margin: 0; font-size: 1.1em;">
       Advancing clinical research through innovative data science tools
     </p>
   </div>

**Development Team:**

* `Moritz Thiele <moritz.thiele@charite.de>`_ 
* `Pedram Ramezani <pedram.ramezani@charite.de>`_ 
* `Noel Kronenberg <noel.kronenberg@charite.de>`_ 
* `Julian Felber <julian.felber@charite.de>`_ 
* `Sophie Hollerbach <sophie.hollerbach@charite.de>`_ 
* `Dario von Wedel <dario.von-wedel@charite.de>`_ 

...and the entire **CORR team** 🙏

.. admonition:: 📊 **Project Stats**
   :class: note

   * **🏥 Version**: |release|
   * **📅 Active Development**: Since September 2024
   * **📈 Publications**: 10+ active projects pending publication
   * **👥 Users**: Research teams across Charité departments with a focus on critical care outcomes research
   * **🔗 GitHub**: `CORR-Vars Repository <https://github.com/cub-corr/corr-vars>`_

.. .. admonition:: 📄 **Citation**
..    :class: tip

..    If you use CORR-Vars in your research, please cite:
   
..    *"Research powered by CORR-Variables, developed by the CORR team at Charité – Universitätsmedizin Berlin"*