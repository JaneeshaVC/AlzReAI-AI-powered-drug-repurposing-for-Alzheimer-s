import pickle
import os
import logging
import torch

# Configure logging
logger = logging.getLogger(__name__)

def load_model_components(model_path='model_components.pkl'):
    """
    Load saved model components with error handling
    
    Args:
        model_path: Path to the saved model components file
        
    Returns:
        dict: Dictionary containing model components or None if failed
    """
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        
        # Check file size
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        logger.info(f"Loading model components from {model_path} ({file_size:.2f} MB)")
        
        # Load the model components
        with open(model_path, 'rb') as f:
            components = pickle.load(f)
        
        # Validate components
        required_keys = ['model', 'hetero_data', 'entity_to_id', 'filtered_node_types']
        missing_keys = [key for key in required_keys if key not in components]
        
        if missing_keys:
            logger.error(f"Missing required components: {missing_keys}")
            return None
        
        # Set model to evaluation mode and CPU
        model = components['model']
        model.eval()
        model.to(torch.device('cpu'))  # Force CPU for Render.com
        
        # Log component information
        logger.info(f"Model loaded successfully:")
        logger.info(f"  - Entity count: {len(components['entity_to_id'])}")
        logger.info(f"  - Node types: {list(components['hetero_data'].node_types)}")
        logger.info(f"  - Edge types: {len(components['hetero_data'].edge_types)}")
        logger.info(f"  - Filtered node types: {len(components['filtered_node_types'])}")
        
        return components
        
    except Exception as e:
        logger.error(f"Error loading model components: {e}")
        return None

def verify_model_components(components):
    """
    Verify that loaded components are valid and functional
    
    Args:
        components: Dictionary containing model components
        
    Returns:
        bool: True if components are valid, False otherwise
    """
    try:
        if not components:
            return False
        
        # Check model
        model = components['model']
        if not hasattr(model, 'forward'):
            logger.error("Model doesn't have forward method")
            return False
        
        # Check hetero_data
        hetero_data = components['hetero_data']
        if not hasattr(hetero_data, 'x_dict') or not hasattr(hetero_data, 'edge_index_dict'):
            logger.error("Hetero data missing required attributes")
            return False
        
        # Check mappings
        entity_to_id = components['entity_to_id']
        filtered_node_types = components['filtered_node_types']
        
        if not isinstance(entity_to_id, dict) or not isinstance(filtered_node_types, dict):
            logger.error("Entity mappings are not dictionaries")
            return False
        
        if len(entity_to_id) == 0 or len(filtered_node_types) == 0:
            logger.error("Empty entity mappings")
            return False
        
        # Test model inference (quick check)
        try:
            model.eval()
            with torch.no_grad():
                embeddings = model(hetero_data.x_dict, hetero_data.edge_index_dict)
                if not embeddings:
                    logger.error("Model inference returned empty results")
                    return False
        except Exception as e:
            logger.error(f"Model inference test failed: {e}")
            return False
        
        logger.info("Model components verification passed")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying model components: {e}")
        return False

def get_model_info(components):
    """
    Get information about loaded model components
    
    Args:
        components: Dictionary containing model components
        
    Returns:
        dict: Information about the model
    """
    if not components:
        return {"status": "not_loaded"}
    
    try:
        info = {
            "status": "loaded",
            "entity_count": len(components['entity_to_id']),
            "node_types": list(components['hetero_data'].node_types),
            "edge_types_count": len(components['hetero_data'].edge_types),
            "filtered_node_types_count": len(components['filtered_node_types']),
            "model_type": str(type(components['model']).__name__)
        }
        
        # Get node counts per type
        hetero_data = components['hetero_data']
        node_counts = {}
        for node_type in hetero_data.node_types:
            node_counts[node_type] = hetero_data[node_type].num_nodes
        info["node_counts"] = node_counts
        
        # Check for specific entities
        entity_to_id = components['entity_to_id']
        alzheimer_id = "Disease::MESH:D000544"
        info["alzheimer_in_graph"] = alzheimer_id in entity_to_id
        
        # Count compounds
        compound_count = sum(1 for entity in entity_to_id.keys() if entity.startswith('Compound::'))
        info["compound_count"] = compound_count
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {"status": "error", "error": str(e)}

# For testing purposes
if __name__ == "__main__":
    print("Testing model loading...")
    
    # Load components
    components = load_model_components()
    
    if components:
        print("✓ Model components loaded successfully")
        
        # Verify components
        if verify_model_components(components):
            print("✓ Model components verification passed")
            
            # Get model info
            info = get_model_info(components)
            print("Model Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print("✗ Model components verification failed")
    else:
        print("✗ Failed to load model components")