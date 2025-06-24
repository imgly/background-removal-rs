---
name: generate_dadjoke
description: Generate a random dad joke about a specific topic
schema:
  type: object
  properties:
    topic:
      type: string
      description: The topic or theme for the dad joke
      default: "programming"
      examples:
        - "programming"
        - "developers"
        - "designers"
        - "cooking"
        - "cats"
  required: []
  additionalProperties: false
tags:
  - humor
  - entertainment
  - dad-jokes
version: "1.0.0"
---

# Generate a random dad joke
Generate a random dad joke about ${topic}. Make it groan-worthy.