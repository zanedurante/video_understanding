### video modules
A video module is best understood as a framework for training on specific tasks.  So, for each task, a different module will be created.  

One minor difference, is that modules have an explicit form for solving the task.  For video captioning, this may look like the general system of:
`video --> video_encoder --> text_decoder --> caption`.  This hides lots of details (which video encoder, which text decoder, which layers are frozen, etc.)  The video module therefore defines the overall structure of the submodules and how they work together to predict the target.