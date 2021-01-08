# Drainage

## Overview

```mermaid
graph LR
    In1(DEM) --> A
    
    subgraph Rasters
        A[precondition<br>resolve flats] --> B
        B[caclulate<br>flow] --> C
    end
        
    subgraph Network
        C[accumulate] --> D[vectorize]
    end
    
    B --> Out1(flow<br>direction)
    C --> Out2(drainage<br>area)
    D --> Out3(stream<br>network)
```

::: workflows.DrainageRasters
    handler: python
    selection:
        members:
            - DrainageRastersWorkflow
    rendering:
        show_root_heading: false
        show_root_toc_entry: false
        show_source: true

::: workflows.DrainageNetwork
    handler: python
    selection:
        members:
            - DrainageNetworkWorkflow
    rendering:
        show_root_heading: false
        show_root_toc_entry: false
        show_source: true

## Operations

### Drainage rasters

::: workflows.DrainageRasters
    handler: python
    selection:
        filters:
            - "!^.*Workflow$"
    rendering:
        show_root_heading: false
        show_root_toc_entry: false
        heading_level: 4
        show_source: true

### Drainage network

::: workflows.DrainageNetwork
    handler: python
    selection:
        filters:
            - "!^.*Workflow$"
    rendering:
        show_root_heading: false
        show_root_toc_entry: false
        heading_level: 4
        show_source: true