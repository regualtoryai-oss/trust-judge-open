# License scope and third-party notices

The MIT license in this repository covers the original application code,
integration adapters, synthetic fixtures and documentation maintained by
`regualtoryai-oss`. This public edition was prepared from a private development
snapshot; the private repository and its history are not part of this release.

Third-party packages are installed through `uv.lock` and
`web/package-lock.json`. They retain their own licenses and notices. This project
does not relicense LangGraph, LangChain, FastAPI, Vue, Vue Flow, DeepSeek Harness,
or any model service, weights or tokenizer assets. No provider credentials,
model weights or third-party SDK source trees are distributed here.

The optional `integrations/dsh/` directory contains application-specific
adapters and synthetic test transports. Its `runtime/package-lock.json` points
to separately installed upstream packages. Installing or using an external SDK
is subject to that SDK's own terms.

The files in `fixtures/import_samples/` are original synthetic examples for
engineering verification. `example.invalid` URLs are intentional placeholders.
Original competition documents, organizer reference annotations, historical
model responses and evaluation archives are excluded. References to optional
external samples in development tools do not grant redistribution rights to
those samples.

No client records, private user accounts, runtime databases or private
configuration are intended to be included. Please report suspected accidental
inclusions privately through the repository owner's GitHub profile; do not post
credentials in a public issue.
