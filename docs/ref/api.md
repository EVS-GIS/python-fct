# API

::: fct.config.Configuration
    handler: python
    selection:
        members:
            - Configuration
            - Workspace
            - DataSource
            - Dataset
            - Tileset
            - FileParser
    rendering:
      show_root_full_path: false
      show_root_members_full_path: true
      show_root_heading: true
      show_root_toc_entry: false
      show_category_heading: true
      show_source: false

::: fct.config.descriptors
    handler: python
    rendering:
      show_root_full_path: false
      show_root_members_full_path: true
      show_root_heading: true
      show_root_toc_entry: false
      show_category_heading: true
      show_source: false

::: workflows.decorators
    handler: python
    rendering:
      show_root_heading: true
      show_object_full_path: true
      show_source: true