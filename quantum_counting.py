import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import cmath

class QuantumCounting:
    def __init__(self, n_qubits, marked_items):
        """
        Initialize quantum counting algorithm
        
        Args:
            n_qubits: Number of qubits for the search space (N = 2^n_qubits items)
            marked_items: List of indices of marked items
        """
        self.n_qubits = n_qubits
        self.N = 2**n_qubits  # Total number of items
        self.marked_items = set(marked_items)
        self.M = len(marked_items)  # Number of marked items
        self.theta = 2 * np.arcsin(np.sqrt(self.M / self.N))  # Theoretical angle
        
    def create_oracle(self):
        """Create the oracle operator that flips the phase of marked items"""
        oracle = np.eye(self.N, dtype=complex)
        for item in self.marked_items:
            oracle[item, item] = -1
        return oracle
    
    def create_diffusion_operator(self):
        """Create the diffusion operator (inversion about average)"""
        # |s⟩ = (1/√N) Σ|x⟩ - uniform superposition
        s_state = np.ones(self.N) / np.sqrt(self.N)
        # Diffusion operator: 2|s⟩⟨s| - I
        diffusion = 2 * np.outer(s_state, s_state) - np.eye(self.N)
        return diffusion
    
    def create_grover_operator(self):
        """Create the Grover operator Q = -Diffusion * Oracle"""
        oracle = self.create_oracle()
        diffusion = self.create_diffusion_operator()
        return -diffusion @ oracle
    
    def quantum_phase_estimation(self, precision_qubits=4):
        """
        Perform quantum phase estimation to find eigenvalues of Grover operator
        
        Args:
            precision_qubits: Number of qubits for phase precision
        """
        grover_op = self.create_grover_operator()
        
        # Find eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eig(grover_op)
        
        # Initialize uniform superposition state
        init_state = np.ones(self.N) / np.sqrt(self.N)
        
        # Project initial state onto eigenvectors
        overlaps = []
        phases = []
        
        for i, (eigenval, eigenvec) in enumerate(zip(eigenvals, eigenvecs.T)):
            overlap = np.abs(np.vdot(eigenvec, init_state))**2
            if overlap > 1e-10:  # Only consider significant overlaps
                phase = np.angle(eigenval) / (2 * np.pi)
                if phase < 0:
                    phase += 1  # Normalize to [0, 1)
                overlaps.append(overlap)
                phases.append(phase)
        
        return phases, overlaps
    
    def estimate_marked_items(self, precision_qubits=4):
        """
        Estimate the number of marked items using quantum counting
        
        Returns:
            estimated_M: Estimated number of marked items
            confidence: Confidence in the estimate
        """
        phases, overlaps = self.quantum_phase_estimation(precision_qubits)
        
        # The phase φ is related to θ by: φ = θ/(2π) or φ = (2π - θ)/(2π)
        # where θ = 2*arcsin(√(M/N))
        
        estimates = []
        confidences = []
        
        for phase, overlap in zip(phases, overlaps):
            # Convert phase back to angle
            theta_est1 = 2 * np.pi * phase
            theta_est2 = 2 * np.pi * (1 - phase)
            
            # Convert angle to number of marked items
            if 0 <= theta_est1 <= np.pi:
                sin_half_theta = np.sin(theta_est1 / 2)
                M_est1 = int(round(self.N * sin_half_theta**2))
                estimates.append(M_est1)
                confidences.append(overlap)
            
            if 0 <= theta_est2 <= np.pi and abs(theta_est2 - theta_est1) > 1e-6:
                sin_half_theta = np.sin(theta_est2 / 2)
                M_est2 = int(round(self.N * sin_half_theta**2))
                estimates.append(M_est2)
                confidences.append(overlap)
        
        # Return the estimate with highest confidence
        if estimates:
            best_idx = np.argmax(confidences)
            return estimates[best_idx], confidences[best_idx]
        else:
            return 0, 0
    
    def simulate_grover_iterations(self, max_iterations=None):
        """
        Simulate Grover's algorithm iterations to show the oscillation
        """
        if max_iterations is None:
            max_iterations = int(np.pi * np.sqrt(self.N) / 2)
        
        grover_op = self.create_grover_operator()
        
        # Start with uniform superposition
        state = np.ones(self.N, dtype=complex) / np.sqrt(self.N)
        
        probabilities = []
        iterations = []
        
        for i in range(max_iterations + 1):
            # Calculate probability of measuring a marked item
            marked_prob = sum(abs(state[j])**2 for j in self.marked_items)
            probabilities.append(marked_prob)
            iterations.append(i)
            
            if i < max_iterations:
                # Apply Grover operator
                state = grover_op @ state
        
        return iterations, probabilities
    
    def visualize_results(self):
        """Visualize the quantum counting results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Grover iterations
        iterations, probabilities = self.simulate_grover_iterations()
        ax1.plot(iterations, probabilities, 'b-', linewidth=2, label='Actual')
        
        # Theoretical curve
        theoretical_prob = [(np.sin((2*i + 1) * self.theta/2))**2 for i in iterations]
        ax1.plot(iterations, theoretical_prob, 'r--', linewidth=2, label='Theoretical')
        
        ax1.set_xlabel('Grover Iterations')
        ax1.set_ylabel('Probability of Marked Item')
        ax1.set_title('Grover Algorithm Oscillation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Phase estimation
        phases, overlaps = self.quantum_phase_estimation()
        ax2.stem(phases, overlaps, basefmt=' ')
        ax2.set_xlabel('Phase (fraction of 2π)')
        ax2.set_ylabel('Overlap Probability')
        ax2.set_title('Quantum Phase Estimation')
        ax2.grid(True, alpha=0.3)
        
        # 3. Eigenvalue distribution
        grover_op = self.create_grover_operator()
        eigenvals = np.linalg.eigvals(grover_op)
        angles = np.angle(eigenvals)
        ax3.scatter(np.real(eigenvals), np.imag(eigenvals), alpha=0.7, s=50)
        ax3.set_xlabel('Real Part')
        ax3.set_ylabel('Imaginary Part')
        ax3.set_title('Eigenvalues of Grover Operator')
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        
        # Add unit circle
        theta_circle = np.linspace(0, 2*np.pi, 100)
        ax3.plot(np.cos(theta_circle), np.sin(theta_circle), 'k--', alpha=0.5)
        
        # 4. Counting accuracy vs precision
        precisions = range(2, 7)
        estimates = []
        confidences = []
        
        for p in precisions:
            est, conf = self.estimate_marked_items(p)
            estimates.append(est)
            confidences.append(conf)
        
        ax4.plot(precisions, estimates, 'bo-', linewidth=2, markersize=8, label='Estimates')
        ax4.axhline(y=self.M, color='r', linestyle='--', linewidth=2, label=f'True value ({self.M})')
        ax4.set_xlabel('Precision Qubits')
        ax4.set_ylabel('Estimated Number of Marked Items')
        ax4.set_title('Counting Accuracy vs Precision')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Example usage and demonstration
def demonstrate_quantum_counting():
    """Demonstrate the quantum counting algorithm"""
    print("Quantum Counting Algorithm Demonstration")
    print("=" * 50)
    
    # Example 1: Small search space
    n_qubits = 4  # 16 items total
    marked_items = [3, 7, 11, 15]  # 4 marked items
    
    qc = QuantumCounting(n_qubits, marked_items)
    
    print(f"Search space: {qc.N} items")
    print(f"Marked items: {marked_items}")
    print(f"True number of marked items: {qc.M}")
    print(f"Theoretical theta: {qc.theta:.4f} radians")
    print()
    
    # Perform quantum counting
    for precision in [3, 4, 5]:
        estimated_M, confidence = qc.estimate_marked_items(precision)
        print(f"Precision qubits: {precision}")
        print(f"Estimated marked items: {estimated_M}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Error: {abs(estimated_M - qc.M)}")
        print()
    
    # Show phases found
    phases, overlaps = qc.quantum_phase_estimation(4)
    print("Phase estimation results:")
    for i, (phase, overlap) in enumerate(zip(phases, overlaps)):
        print(f"Phase {i+1}: {phase:.4f} (overlap: {overlap:.4f})")
    print()
    
    # Visualize results
    qc.visualize_results()
    
    # Example 2: Different ratio
    print("\nExample 2: Different marked item ratio")
    print("=" * 40)
    
    marked_items_2 = [1, 5]  # 2 marked items out of 16
    qc2 = QuantumCounting(n_qubits, marked_items_2)
    
    print(f"Marked items: {marked_items_2}")
    print(f"True number of marked items: {qc2.M}")
    
    estimated_M2, confidence2 = qc2.estimate_marked_items(4)
    print(f"Estimated marked items: {estimated_M2}")
    print(f"Confidence: {confidence2:.4f}")
    print(f"Error: {abs(estimated_M2 - qc2.M)}")

if __name__ == "__main__":
    demonstrate_quantum_counting()