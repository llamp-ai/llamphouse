import yaml

with open("openapi.yml", "r") as f:
    spec = yaml.safe_load(f)

paths = spec.get("paths", {})

# only keep all the paths that start with /assistants or /threads
filtered_paths = {
    k: v for k, v in paths.items() if (k.startswith("/assistants") or k.startswith("/threads"))
}

# get all the component references that are used in the filtered paths
# for path, methods in filtered_paths.items():
#     for method, details in methods.items():
#         if "requestBody" in details and "content" in details["requestBody"]:
#             content = details["requestBody"]["content"]
#             for media_type, media_details in content.items():
#                 if "schema" in media_details:
#                     schema = media_details["schema"]
#                     if "$ref" in schema:
#                         ref = schema["$ref"]
#                         # Here you can process the reference if needed
#                         print(f"Found reference: {ref} in path {path} method {method}")

# Save the filtered paths to a new YAML file keeping the original structure
filtered_spec = {
    "openapi": "3.1.0",
    "info": spec.get("info", {}),
    "servers": spec.get("servers", []),
    "security": spec.get("security", []),
    "tags": [
        {
            "name": "assistant",
            "description": "Operations related to assistants"
        }
    ],
    "paths": filtered_paths,
    "components": spec.get("components", {})
}
with open("filtered_openapi.yml", "w") as f:
    yaml.safe_dump(filtered_spec, f, default_flow_style=False)