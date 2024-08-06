def load_config(dataset_key="test", verbose=False):
    """Loads dictionary with path names and any other variables set through config.toml

    Args:
        verbose (bool, optional): print paths

    Returns:
        config: dict
    """
    from pathlib import Path

    import toml

    package_dir = Path(__file__).parent.parent.parent.absolute()
    config_file = package_dir / "config.toml"

    if verbose:
        print(f"package dir: {package_dir}")
        print(f"config file: {config_file}")

    config = dict()
    toml_dict = toml.load(config_file)
    config.update(toml_dict[dataset_key])
    config.update({"package_dir": package_dir, "config_file": config_file})
    not_found = []
    for key in config:
        if Path(config[key]).exists():
            config[key] = Path(config[key])

    for key in not_found:
        print(f"Did not find {key}")
        config.pop(key, None)

    return config
