"""Test that load_config supports both .yml and .yaml file extensions."""

from pathlib import Path
import pytest
from dynx.stagecraft.io import load_config


def test_load_config_with_yaml_extension(tmp_path: Path):
    """Test that load_config works with .yaml file extensions."""
    # Create config directory structure
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    
    # Create master.yml
    (cfg_dir / "master.yml").write_text("name: TestModel\nhorizon: 2\n")
    
    # Create connections.yml
    (cfg_dir / "connections.yml").write_text("intra_period:\n  forward: []\n")
    
    # Create stages directory with mixed .yml and .yaml files
    stages_dir = cfg_dir / "stages"
    stages_dir.mkdir()
    
    # Create some stages with .yml extension
    (stages_dir / "Stage0.yml").write_text("name: Stage0\ntype: basic\n")
    (stages_dir / "Stage1.yml").write_text("name: Stage1\ntype: basic\n")
    
    # Create some stages with .yaml extension
    (stages_dir / "Stage2.yaml").write_text("name: Stage2\ntype: advanced\n")
    (stages_dir / "Stage3.yaml").write_text("name: Stage3\ntype: advanced\n")
    
    # Load the configuration
    config = load_config(cfg_dir)
    
    # Verify all files were loaded
    assert "master" in config
    assert "stages" in config
    assert "connections" in config
    
    # Verify all stages were loaded regardless of extension
    assert len(config["stages"]) == 4
    assert "STAGE0" in config["stages"]
    assert "STAGE1" in config["stages"]
    assert "STAGE2" in config["stages"]
    assert "STAGE3" in config["stages"]
    
    # Verify content was loaded correctly
    assert config["stages"]["STAGE0"]["type"] == "basic"
    assert config["stages"]["STAGE2"]["type"] == "advanced"


def test_load_config_no_stage_files(tmp_path: Path):
    """Test that load_config raises error when no stage files are found."""
    # Create config directory structure
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    
    # Create master.yml and connections.yml
    (cfg_dir / "master.yml").write_text("name: TestModel\n")
    (cfg_dir / "connections.yml").write_text("intra_period: {}\n")
    
    # Create empty stages directory
    stages_dir = cfg_dir / "stages"
    stages_dir.mkdir()
    
    # Should raise error since no stage files exist
    with pytest.raises(FileNotFoundError, match="No stage configuration files found"):
        load_config(cfg_dir)


def test_load_config_lowercase_names_uppercased(tmp_path: Path):
    """Test that lowercase stage filenames are properly uppercased."""
    # Create config directory structure
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    
    # Create master.yml and connections.yml
    (cfg_dir / "master.yml").write_text("name: TestModel\n")
    (cfg_dir / "connections.yml").write_text("intra_period: {}\n")
    
    # Create stages directory with lowercase filenames
    stages_dir = cfg_dir / "stages"
    stages_dir.mkdir()
    
    # Create stages with lowercase names
    (stages_dir / "ownh.yml").write_text("name: ownh\ntype: ownership\n")
    (stages_dir / "tenu.yaml").write_text("name: tenu\ntype: tenure\n")
    (stages_dir / "mixed_Case.yml").write_text("name: mixed_Case\ntype: mixed\n")
    
    # Load configuration
    config = load_config(cfg_dir)
    
    # Verify all stage names are uppercased
    assert "OWNH" in config["stages"]
    assert "TENU" in config["stages"]
    assert "MIXED_CASE" in config["stages"]
    
    # Verify lowercase versions are NOT present
    assert "ownh" not in config["stages"]
    assert "tenu" not in config["stages"]
    assert "mixed_Case" not in config["stages"]
    
    # Verify content is preserved
    assert config["stages"]["OWNH"]["type"] == "ownership"
    assert config["stages"]["TENU"]["type"] == "tenure"
    assert config["stages"]["MIXED_CASE"]["type"] == "mixed" 