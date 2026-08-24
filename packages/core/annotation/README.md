# `@swarmx/annotation`

Provider-neutral annotation contract shared by conversation UI, Science UI, and Host tools.

`Annotation` is a strict superset of the current OpenAI Responses `output_text.annotations`
union. The official `file_citation`, `url_citation`, `container_file_citation`, and `file_path`
objects are accepted without an envelope or renamed fields, and provider-added fields on those
known variants survive parsing. SwarmX adds one separate `comment` branch whose target is
`document_text`, `document_region`, or `image_point`.

OpenAI output citations and user-authored comments remain different semantic branches. A consumer
must switch on `type`; it must not reinterpret an OpenAI citation as a SwarmX comment or add
required SwarmX fields to an OpenAI object.
