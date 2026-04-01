# Requirements for experiment & experiment runs behavior: 

1. The base webapp loaded should be assigned project=1 & user=1
2. When the webapp is first loaded it should be opened to the last loaded experiment & experiment run 
3. Creating new training experiment runs: 
    a. New training experiment runs are created everytime the user starts training by clicking on `Start training` CTA in training mode. 
    b. When a new experiment is created this way, load this new experiment & run 
    c. New training experiments runs are also created when the user explicitly creates a new run from the Experiments tab > Experiment runs view. 
    d. Add a new CTA button here called `Create & load new run` to be able to do this
    e. When this CTA is clicked automatically create & switch to this new experiment run under the experiment which is currently active. 
    f. On the server side, also ensure that the new training experiment run is added to the DB.
    g. Also ensure that the run data is regularly being updated to the DB at appropriate frequencies, even if the run has not finished yet.
4. Creating a new training experiment: 
    a. Users should also be able to create a experiment itself via the Experiments tab’s base page: 
    b. Add a CTA button called `Create & load new experiment` to be able to do this. 
    c. When this CTA is clicked automatically create & load this new training experiment, keep the run empty for now. 
    d. The newly loaded experiment ID should be visible via the breadcrumb UI component as well. 
    e. On the server side, also ensure that the new training experiment is added to the DB. 
    f. When a new run is added, ensure that it is linked with the experiment appropriately in the DB. 
5. Loading experiment runs: 
    a. The user should be able to load any experiment run via the Experiments tab > Runs view. 
    b. When done so if the existing experiment run was already executing then it should continue running in the background.
    c. Load this newly requested run load this run’s model, reward function, training graph, metrics, logs in the respective tabs & logger component. 
    d. Once loaded, the run ID should be updated in the breadcrumb component as well.
6. Editing currently loaded experiment run: 
    a. If a user edits any part of the training graph or reward function, it should result in the `Start Training` or `Pause Training` CTA to basically have a dropdown the right edge using which the user can launch a new training run, label: `Start new run`
    b. If the user uses this, create a new run, load this run as the current active run & start training for this run as well.

This needs to be implemented for the app backend server & the client. 

Now check the instructions/specs/experiments.md file. This provides an implementation PRD for the training experiments & experiment runs layer of the product. I want you to implement this for the app backend server & the client. Plan this carefully before hand on all the changes that need to be made on both sides. Use the existing ground work already done in terms of the UI on the client side & the orms added on the backend server. 
