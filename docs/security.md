# Security and Limitations

- **Local single-user demo only**. No authentication, authorization, user isolation, quotas or rate limits.
- Uploads are capped at 20 MB, have server-generated filenames, and are checked with PyMuPDF. This does not sandbox PDF parsing or prevent resource exhaustion.
- Uploaded PDF text is sent to the configured LLM provider when answering questions. Do not upload sensitive material without reviewing data controls.
- Retrieved text is untrusted and can contain prompt injection attempts.
- Index rebuilding is serialized but not transactional; failures may require a manual rebuild.
- The API health endpoint is liveness only, not full dependency readiness.
- `.env`, PDF data, and local indexes are excluded from Git and Docker build contexts.
- For public hosting, add auth, tenant isolation, storage quotas, HTTPS, backups, structured logging, limits, and transactional indexing.
