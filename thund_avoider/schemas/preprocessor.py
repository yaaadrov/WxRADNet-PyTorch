from pydantic import BaseModel


class RasterioCompressionSchema(BaseModel):
    compress: str = "lzw"
    tiled: bool = True
    blockxsize: int = 256
    blockysize: int = 256
    predictor: int = 2
