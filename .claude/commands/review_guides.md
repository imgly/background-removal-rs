---
name: review_guides
description: Comprehensive testing and review of developer guides by following all steps and validating functionality
schema:
  type: object
  properties:
    url:
      type: string
      format: uri
      description: The URL of the developer guide to test and review
      examples:
        - "https://docs.example.com/getting-started"
        - "https://guides.framework.com/quickstart"
        - "https://tutorial.library.com/setup"
  required: ["url"]
  additionalProperties: false
tags:
  - testing
  - guides
  - validation
  - developer-experience
  - qa
version: "1.0.0"
---

# Developer Guides Review Command

Go through all steps explained at ${url}. 

Take all instructions exactly as described from the documentation pages.
Check the package json whenever you encounter an error.
Please do all the work in a temporary subdirectory inside "tmp". 

Verify that all works by running the app in the background and testing it with playwright mcp or cli.

If all is done cleanup all temporary files.

Create a summary with all smaller errors and where they can be found and how to fix them.
If all worked then print ✅, with minor errors print 💛 else 🛑.

Analyze the guide for best practices.

Do a diligence of the code you see and add it to potential improvements with explainer.
Add the URL as info so that I know which guide you tested.

Write the summary into a file as task result into the reports directory named like the task and file should have a unique filename like 'report_0001.md' and counting up and not overwrite any existing files.

