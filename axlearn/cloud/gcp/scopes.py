# Copyright © 2023 Apple Inc.

"""Pre-defined scopes for gcloud credentials.

See: https://developers.google.com/identity/protocols/oauth2/scopes
"""

_OPEN_ID = "openid"  # Google's OAuth 2.0 API for OpenID Connect.

# Detailed documentation and permissions for each scope can be found on
# https://developers.google.com/identity/protocols/oauth2/scopes
_CLOUD_PLATFORM = "https://www.googleapis.com/auth/cloud-platform"  # general GCP API call
_CLOUD_TPU = "https://www.googleapis.com/auth/cloud.tpu"  # TPU operations
_COMPUTE = "https://www.googleapis.com/auth/compute"  # GCE operations
# Read primary email address of token owner.
_EMAIL = "https://www.googleapis.com/auth/userinfo.email"
_SQL_LOGIN = "https://www.googleapis.com/auth/sqlservice.login"  # CloudSQL login
_STORAGE_RW = "https://www.googleapis.com/auth/devstorage.read_write"  # GCS Read-Write


# For typical TPU node and queued resource operations, including list, create, delete etc.
# See: https://cloud.google.com/tpu/docs/reference/rest
# TODO(Zhaoyi): create more granular scopes for TPU operations.
DEFAULT_TPU_SCOPES = [_CLOUD_TPU, _CLOUD_PLATFORM]

# Same scopes used by gcloud auth application-default login.
# based on https://cloud.google.com/sdk/gcloud/reference/auth/application-default/login
DEFAULT_APPLICATION = [_OPEN_ID, _EMAIL, _CLOUD_PLATFORM, _SQL_LOGIN]
