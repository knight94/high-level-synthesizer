# High-level-synthesizer
Simplied version of Digital HLS <br />
Scheduler: List scheduling is used for both minimum latency and minimum resource <br />
Binder: Resource binding is done using clique partitioning and register binding is done via left edge algorithm <br />
Datapath: Muxes are used to interconnect registers and functional units <br />
Work is done by Naman Jain (IIT Delhi) as part of COL 719: Synthesis of digital systems Semester I (2021-22)

# Dependencies
Python Graphviz package

# Control flow graph
![](Output_images/gcd_HDL_cfg.png)
![](Output_images/modulo_HDL_cfg.png)

# Control and Data flow graph
![](Output_images/gcd_HDL_cdfg.png)
![](Output_images/modulo_HDL_cdfg.png)

# Example block interval graph
![](Output_fimages/block_4_interval.svg)

# Datapath
# Minimum latency
![](Output_images/gcd_HDLdatapath_min_latency.png)
![](Output_images/modulo_HDLdatapath.png)

# Minimum resorce
![](Output_images/gcd_HDLdatapath_MIN_r.png)

# FSM for controller
# Minimum latency
![](Output_images/gcd_HDL_fsm_min_latency.png)
![](Output_images/modulo_HDL_fsm.png)

# Minimum resorce
![](Output_images/gcd_HDL_fsm_min_r.png)
