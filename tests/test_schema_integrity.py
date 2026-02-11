from src.schema import MediaSource


def test_mediasource_structure_regression(data_regression):
    """
    Captures the schema structure. If you change MediaSource,
    this will fail until you run 'pytest --force-regen'.
    """
    # Create a dummy instance with all fields populated
    sample_data = {"uri": "placeholder", "media_type": "video", "source_type": "local"}
    source = MediaSource(**sample_data)

    # data_regression converts the object to a dict and compares it
    # to a saved YAML file in tests/test_schema_integrity/
    data_regression.check(source.model_dump())
