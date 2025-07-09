#!/usr/bin/env python3
"""
Example: Grasp detection with ElasticROS
Demonstrates elastic computing for robotic grasping tasks
"""

import numpy as np
import time
import cv2
from typing import Dict, List, Tuple

# ElasticROS imports
from elasticros_core import ElasticNode, PressNode, ReleaseNode


class GraspDetectionPressNode(PressNode):
    """Local computation for grasp detection"""
    
    def _initialize(self):
        """Initialize local processing resources"""
        # In real implementation, might load lightweight preprocessing models
        self.input_size = (640, 480)
        self.preprocessing_mean = np.array([0.485, 0.456, 0.406])
        self.preprocessing_std = np.array([0.229, 0.224, 0.225])
        
    def _process(self, data: np.ndarray, compute_ratio: float) -> Dict:
        """
        Process image for grasp detection.
        
        Args:
            data: Input image
            compute_ratio: How much to process locally
            
        Returns:
            Processed data or intermediate result
        """
        if compute_ratio == 0.0:
            # No local processing
            return {'raw_image': data}
            
        # Step 1: Resize (always done locally if any processing)
        if data.shape[:2] != self.input_size:
            resized = cv2.resize(data, self.input_size)
        else:
            resized = data
            
        if compute_ratio <= 0.3:
            # Minimal processing - just resize
            return {
                'processed_image': resized,
                'preprocessing_level': 'resize_only'
            }
            
        # Step 2: Color space conversion and normalization
        if len(resized.shape) == 2:
            # Grayscale to RGB
            normalized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            normalized = resized
            
        # Normalize
        normalized = normalized.astype(np.float32) / 255.0
        normalized = (normalized - self.preprocessing_mean) / self.preprocessing_std
        
        if compute_ratio <= 0.7:
            # Partial processing
            return {
                'processed_image': normalized,
                'preprocessing_level': 'normalized'
            }
            
        # Step 3: Edge detection and feature extraction (full local)
        # Simplified - in practice might use more sophisticated features
        edges = cv2.Canny((resized * 255).astype(np.uint8), 50, 150)
        
        # Extract regions of interest
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get top regions by area
        regions = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            x, y, w, h = cv2.boundingRect(contour)
            regions.append({
                'bbox': [x, y, w, h],
                'area': cv2.contourArea(contour)
            })
            
        return {
            'processed_image': normalized,
            'preprocessing_level': 'full',
            'regions_of_interest': regions,
            'edges': edges
        }


class GraspDetectionReleaseNode(ReleaseNode):
    """Cloud computation for grasp detection using deep learning"""
    
    def _initialize(self):
        """Initialize cloud resources and models"""
        # Simulate loading a heavy ML model
        print("Loading grasp detection model in cloud...")
        time.sleep(1.0)  # Simulate model loading
        
        # In practice, would load actual model like:
        # self.model = load_grasp_detection_model()
        self.model_loaded = True
        
    def _process(self, data: Dict, compute_ratio: float) -> Dict:
        """
        Complete grasp detection in cloud.
        
        Args:
            data: Input from Press node
            compute_ratio: Cloud computation ratio
            
        Returns:
            Grasp detection results
        """
        # Extract data based on preprocessing level
        if 'raw_image' in data:
            # Full cloud processing needed
            image = data['raw_image']
            # Do all preprocessing
            image = cv2.resize(image, (640, 480))
            image = image.astype(np.float32) / 255.0
            
        elif 'processed_image' in data:
            # Use preprocessed image
            image = data['processed_image']
            preprocessing_level = data.get('preprocessing_level', 'unknown')
            
            # Complete any remaining preprocessing
            if preprocessing_level == 'resize_only':
                # Still need normalization
                image = image.astype(np.float32) / 255.0
                
        else:
            raise ValueError("No image data received")
            
        # Run inference (simulated)
        inference_start = time.time()
        
        # Simulate GPU inference
        if self.use_gpu:
            time.sleep(0.05)  # Fast GPU inference
        else:
            time.sleep(0.2)   # Slower CPU inference
            
        # Generate mock grasp predictions
        grasps = self._generate_mock_grasps(image, data.get('regions_of_interest', []))
        
        inference_time = time.time() - inference_start
        
        return {
            'grasps': grasps,
            'inference_time': inference_time,
            'processed_on': self.instance_type,
            'preprocessing_level': data.get('preprocessing_level', 'cloud'),
            'timestamp': time.time()
        }
        
    def _generate_mock_grasps(self, image: np.ndarray, regions: List[Dict]) -> List[Dict]:
        """Generate mock grasp predictions"""
        grasps = []
        
        # If we have regions of interest, use them
        if regions:
            for i, region in enumerate(regions[:5]):  # Top 5 regions
                bbox = region['bbox']
                center_x = bbox[0] + bbox[2] // 2
                center_y = bbox[1] + bbox[3] // 2
                
                grasps.append({
                    'position': [center_x, center_y],
                    'angle': np.random.uniform(-90, 90),
                    'width': min(bbox[2], bbox[3]) * 0.8,
                    'confidence': 0.95 - i * 0.1
                })
        else:
            # Generate random grasps
            h, w = image.shape[:2]
            for i in range(5):
                grasps.append({
                    'position': [
                        np.random.randint(w // 4, 3 * w // 4),
                        np.random.randint(h // 4, 3 * h // 4)
                    ],
                    'angle': np.random.uniform(-90, 90),
                    'width': np.random.randint(30, 80),
                    'confidence': 0.9 - i * 0.15
                })
                
        return sorted(grasps, key=lambda x: x['confidence'], reverse=True)


def main():
    """Main function demonstrating elastic grasp detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ElasticROS Grasp Detection Example')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file')
    parser.add_argument('--image', type=str, default=None,
                       help='Input image path')
    parser.add_argument('--camera', action='store_true',
                       help='Use camera input')
    parser.add_argument('--simulate-bandwidth', type=float, default=None,
                       help='Simulate bandwidth limit (Mbps)')
    args = parser.parse_args()
    
    # Initialize ElasticNode
    elastic_node = ElasticNode(args.config)
    
    # Create and register nodes
    press_node = GraspDetectionPressNode("grasp_press")
    release_node = GraspDetectionReleaseNode(
        "grasp_release",
        config={'instance_type': 'g4dn.xlarge', 'use_gpu': True}
    )
    
    elastic_node.register_node_pair("grasp_detection", press_node, release_node)
    
    # Simulate bandwidth constraint if requested
    if args.simulate_bandwidth:
        from elasticros_core.utils import BandwidthController
        bandwidth_controller = BandwidthController()
        # Convert Mbps to kbps
        bandwidth_kbps = int(args.simulate_bandwidth * 1000)
        bandwidth_controller.set_bandwidth_limit(bandwidth_kbps, bandwidth_kbps)
        print(f"Set bandwidth limit to {args.simulate_bandwidth} Mbps")
    
    # Process images
    if args.camera:
        # Camera input
        cap = cv2.VideoCapture(0)
        print("Press 'q' to quit, 's' to show statistics")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process with ElasticROS
            start_time = time.time()
            result = elastic_node.elastic_execute("grasp_detection", frame)
            total_time = time.time() - start_time
            
            # Visualize results
            vis_frame = frame.copy()
            for grasp in result['grasps'][:3]:  # Top 3 grasps
                x, y = grasp['position']
                angle = grasp['angle']
                width = grasp['width']
                conf = grasp['confidence']
                
                # Draw grasp rectangle
                cv2.circle(vis_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                
                # Draw grasp orientation
                dx = int(width * np.cos(np.radians(angle)) / 2)
                dy = int(width * np.sin(np.radians(angle)) / 2)
                cv2.line(vis_frame, (int(x-dx), int(y-dy)), 
                        (int(x+dx), int(y+dy)), (0, 255, 0), 2)
                
                # Draw confidence
                cv2.putText(vis_frame, f"{conf:.2f}", 
                           (int(x), int(y-10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show timing info
            cv2.putText(vis_frame, f"Total: {total_time*1000:.1f}ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Inference: {result['inference_time']*1000:.1f}ms", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Processed: {result['preprocessing_level']}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Grasp Detection', vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                stats = elastic_node.get_statistics()
                print("\n=== ElasticROS Statistics ===")
                print(f"Average execution time: {stats['average_execution_time']*1000:.1f}ms")
                print(f"Action distribution: {stats['action_distribution']}")
                print(f"Total executions: {stats['total_executions']}")
                print(f"Regret bound: {stats['regret_bound']:.2f}")
                
        cap.release()
        cv2.destroyAllWindows()
        
    else:
        # Single image
        if args.image:
            image = cv2.imread(args.image)
        else:
            # Generate random image
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
        # Process multiple times to see adaptation
        print("Processing image multiple times to demonstrate adaptation...")
        
        for i in range(10):
            start_time = time.time()
            result = elastic_node.elastic_execute("grasp_detection", image)
            total_time = time.time() - start_time
            
            print(f"\nIteration {i+1}:")
            print(f"  Total time: {total_time*1000:.1f}ms")
            print(f"  Inference time: {result['inference_time']*1000:.1f}ms")
            print(f"  Preprocessing: {result['preprocessing_level']}")
            print(f"  Top grasp confidence: {result['grasps'][0]['confidence']:.2f}")
            
            # Simulate bandwidth change
            if i == 5 and args.simulate_bandwidth:
                new_bandwidth = args.simulate_bandwidth / 3
                bandwidth_kbps = int(new_bandwidth * 1000)
                bandwidth_controller.set_bandwidth_limit(bandwidth_kbps, bandwidth_kbps)
                print(f"\n!!! Bandwidth reduced to {new_bandwidth} Mbps !!!\n")
                
        # Final statistics
        stats = elastic_node.get_statistics()
        print("\n=== Final Statistics ===")
        print(f"Average execution time: {stats['average_execution_time']*1000:.1f}ms")
        print(f"Action distribution: {stats['action_distribution']}")
        print(f"Total executions: {stats['total_executions']}")
        
    # Cleanup
    if args.simulate_bandwidth:
        bandwidth_controller.clear_limits()
        print("Cleared bandwidth limits")
        
    elastic_node.shutdown()
    

if __name__ == '__main__':
    main()