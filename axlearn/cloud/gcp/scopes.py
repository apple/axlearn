""" Pre-defined scopes for gcloud credentials.

    See:
        https://developers.google.com/identity/protocols/oauth2/scopes
"""


CLOUD_PLATFORM = "https://www.googleapis.com/auth/cloud-platform"
CLOUD_TPE = "https://www.googleapis.com/auth/cloud.tpu"
COMPUTE = "https://www.googleapis.com/auth/compute"
STORAGE_RW = "https://www.googleapis.com/auth/devstorage.read_write"
OPEN_ID = "openid"
EMAIL = "https://www.googleapis.com/auth/userinfo.email"
SQL_LOGIN = "https://www.googleapis.com/auth/sqlservice.login"


# For typical TPU node and queued resource operations, including list, create, delete etc.
# See: https://cloud.google.com/tpu/docs/reference/rest
DEFAULT_TPU_SCOPES = [CLOUD_TPE, CLOUD_PLATFORM]

# Same scopes used by gcloud auth application-default login
# based on https://cloud.google.com/sdk/gcloud/reference/auth/application-default/login
DEFAULT_APPLICATION = [OPEN_ID, EMAIL, CLOUD_PLATFORM, SQL_LOGIN]
