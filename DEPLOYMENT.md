# Deployment status

DALIL must not yet be deployed as a public production service. Startup intentionally fails on Azure App Service and whenever `APP_ENV=production`, because production identity and durable storage are not implemented and verified. The shared development token and the `LOCAL_DEV_MODE` / `PRIVATE_DEV_MODE` flags are not production authentication.

A release needs, at minimum, validated Microsoft Entra authentication and stable workspace identity, storage that survives restarts, safe single-writer index updates and backups, HTTPS, approved secret configuration, and a real OpenAI upload/query/citation smoke test. Confirm pricing and subscription limits before provisioning resources.

See the README for local setup and staging a separate index. A staged index is not automatically activated; validate retrieval against the source PDF and configure both index and source-document paths only after validation. Preserve the previous index and documents for rollback.
