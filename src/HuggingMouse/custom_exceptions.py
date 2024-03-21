class AllenCachePathNotSpecifiedError(Exception):

    def __init__(self, message=" Allen cache path is not specified as an environment variable! Set HGMS_ALLEN_CACHE_PATH with EXPORT or .env file"):
        self.message = message
        super().__init__(self.message)


class TransformerEmbeddingCachePathNotSpecifiedError(Exception):

    def __init__(self, message="Transformer embedding path cache path is not specified as an environment variable! Set HGMS_TRANSF_EMBEDDING_PATH with EXPORT or .env file"):
        self.message = message
        super().__init__(self.message)


class RegressionOutputCachePathNotSpecifiedError(Exception):

    def __init__(self, message="Regression analysis output path cache path is not specified as an environment variable! Set HGMS_REGR_ANAL_PATH with EXPORT or .env file"):
        self.message = message
        super().__init__(self.message)
