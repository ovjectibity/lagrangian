## Problem statement — Checkpoints: 

1. No checkpoints being saved during local training run on the server.

2. Remote worker saves & sends the checkpoints as training updates, but they are not being stored permanently currently by the server.

3. No way for user to check the existing checkpoints.

4. No way for user to use existing checkpoints within training runs.

## Solution — Checkpoints: 

1. Saving checkpoint updates for a locally ran training workload: 
    a. Skip this for now

2. Saving checkpoint updates from remote worker: 
    a. Allow users to also configure the CheckpointFrequency field in the TrainingJobController. Enable this capability on the client side’s training graph. 
    b. Additionally, 1 more option can be used by the customer: saving only the last checkpoint as added to `commondef/training.rs` > `CheckpointFrequency` . This would basically save only the last checkpoint available in the training job either right before it finishes or even after the user stops the training job. The remote worker needs to be changed to adhere to these options.
    c. When checkpoint updates are sent from the remote worker, they need to be saved by the app server in a local directory (for now only, we’ll use some S3-like server based storage later)
    d. Also 2 DB entries need to be updated: 
        i. training_runs: A new id (reference to the added checkpoint record) added to the checkpoints_refs attribute
        ii. checkpoint_references: A new entry recording the new checkpoint

3. Viewing checkpoints for a run on the client side: 
    a. Add another tab to the editor pane on the client sided named `Checkpoints`
    b. This pane should display a list of checkpoints saved. Use the card design used for the experiment runs view here as well to show checkpoints. Each card should display: 
        i. Checkpoint ID
        ii. Training experiment ID
        iii. Training run ID
        iv. Checkpoint timestamp
    c. As soon as checkpoints are saved by the server, they should be dynamically informed about to the client as well 
        i. Send this update via the TrainingUpdateMessage itself 

4. Using prior checkpoints in a new training run: 
    a. Users should be able to use prior saved checkpoints in subsequent runs 
    b. The way to provide this capability will be via the training graph 
    c. In the control model & value model node, in the type param, apart from FFN another param to be added as `Checkpoint` 
    d. On the user selecting this type, all the other fields should freeze (num_hidden_layers, num_layers, output_dim, input_dim) & an additional field should show up saying `Checkpoint ID` which basically will be a dropdown of all available checkpoint IDs. 
        i. Ensure that the checkpoint IDs listed in the menu bar segregate IDs on the basis the training experiment & run hierarchy with ID labels for both these attributes above every set of checkpoint IDs belonging to them 
        ii. The other fields should automatically populate (but disabled) on the basis of the checkpoint which is loaded. 
    e. In order to do this hence, the client will need GetAllCheckpoints, GetCheckpointInfo messages to send to the server & to which the server will respond. 
    f. Once the user starts a new training run with a checkpoint, ensure to use that instead of a model from scratch. This is to be done for both the local training job run type & the remote training job type. 

5. Saving checkpoints & sending them to remoteworker: 
    a. When the user starts a training run on a remote worker using a checkpoint, it needs to be transferred from the local storage of the app server to the remote worker. Use newly added: `pub checkpoint: Option<Checkpoint>`  in the `commondef/remoteworkermessages.rs` for the same. 
    b. Ensure that checkpoints sent by the remote worker to the server & vice versa are sent via a binary stream in WSS instead of base64 encoded.
    c. On the remote worker side, ensure to extract & load it correctly when starting the training job.