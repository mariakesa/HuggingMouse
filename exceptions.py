class CachePathNotSpecifiedError(Exception):
    """Exception raised when the cache path is not specified in the JSON file."""
    
    def __init__(self, message="Cache path is not specified in the JSON file!"):
        self.message = message
        super().__init__(self.message)