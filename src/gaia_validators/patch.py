def patch_pydantic_v1():
    import pydantic

    if pydantic.__version__ >= "2.0.0":
        raise RuntimeError("Cannot patch pydantic version >= '2.0.2'")

    from pydantic import BaseModel, validator

    def field_validator(*args, **kwargs):
        pre = kwargs.get("mode", "after") == "before"
        check_fields = kwargs.get("check_fields", None)
        return validator(*args, pre=pre, check_fields=check_fields)

    BaseModel.model_dump = BaseModel.dict

    setattr(pydantic, "field_validator", field_validator)
    setattr(pydantic, "BaseModel", BaseModel)
