
/*
# 1. Save code as nn.c
# 2. Compile
gcc -o nn nn.c -lm

# 3. Run examples
./nn 1 1    # Should learn → YES
./nn 1 0    # Should stay → NO
./nn 0 0    # Should stay → NO
./nn 0 1    # Should stay → NO
*/


/*

=== Neural Network Training (AND Logic) ===
Input: [1, 1]
Target output: 1

Initial weights: w1=0.300, w2=0.400, bias=-0.500

BEFORE training: z = 0.200 → output = 0.200

--- Training Start (Backpropagation) ---
Epoch  1: output=0.200, error=0.800 → w1=0.700, w2=0.800, b=-0.100
Epoch  2: output=1.400, error=-0.400 → w1=0.500, w2=0.600, b=-0.300
...
Epoch 10: output=0.900, error=0.100 → ...

--- Training Complete ---
Final z = 1.200 → Final output = 1.200
Final decision: YES

*/

/*
Let’s give you a clear, real-world metaphor so you can see the example 
in your mind every time you run ./nn 1 2.

Metaphor: "Can I open the safe?"
The Safe Has 2 LocksLock
Needs to be...
Key #1 Inserted (1)
Key #2 Inserted (1)
Only if BOTH keys are inserted → Safe opens!

Your Neural Network = The Safe’s BrainIt receives 2 inputs (the keys):

./nn 1 1   → Both keys in → Safe opens → YES
./nn 1 0   → One key missing → Safe stays closed → NO

The Weights = "How important is each key?"
Weight        Meaning
w1            How much Key #1 matters
w2            How much Key #2 matters
b             The threshold — how hard it is to open


Training = Teaching the Safe How to Recognize Both Keys
Step     What Happens
1        You try: ./nn 1 1 → should say YES, but it says NO
2        Backpropagation: "Ah! Both keys were in — I should’ve opened!"
3        Adjust w1, w2, b → make it easier to open when both are 1
4        Repeat → safe learns: only 1+1 opens it


Final Behavior (After Training)
Command    Input    Output    Safe
./nn 1 1   [1,1]    >0        Opens
./nn 1 0   [1,0]    0         Closed
./nn 0 1   [0,1]    0         Closed
./nn 0 0   [0,0]    0         Closed


Your Mental Model (Use This!)
Think of your C program as a tiny brain inside a safe.
It watches two key slots.
Only when both are filled (1,1) → it says "OPEN!"
The weights are like how much it trusts each key.
Backpropagation = teaching it to trust both keys equally.


Bonus: Real-World Uses of This Pattern
Problem        Input 1          Input 2          Output
Loan approval  Has job (1)      Good credit (1)  Approve
Game move      Player near (1)  Enemy weak (1)   Attack
Alarm          Motion (1)       Door open (1)    Sound

Final Summary (Your Metaphor)
./nn 1 1  →  "Both keys in!" → Brain computes → ReLU → Output > 0 → "SAFE OPENS!"
./nn 1 0  →  "Missing a key!" → ReLU blocks → Output = 0 → "STAYS CLOSED"

*/

// =======================================================
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// -----------------------------
// ReLU Activation Function
// -----------------------------
double relu(double z) {
    return (z > 0) ? z : 0;  // If z > 0 → z, else → 0
}

// -----------------------------
// Derivative of ReLU (for backprop)
// -----------------------------
double relu_derivative(double z) {
    return (z > 0) ? 1.0 : 0.0;  // If z > 0 → slope=1, else → slope=0
}

// -----------------------------
// Main Function
// -----------------------------
int main(int argc, char *argv[]) {
    // === 1. Check command line input ===
    if (argc != 3) {
        printf("Usage: %s <input1> <input2>\n", argv[0]);
        printf("Example: %s 1 1\n", argv[0]);
        return 1;
    }

    // === 2. Read two inputs from command line ===
    double x1 = atof(argv[1]);  // First input
    double x2 = atof(argv[2]);  // Second input
    double inputs[2] = {x1, x2};

    printf("=== Neural Network Training (AND Logic) ===\n");
    printf("Input: [%.0f, %.0f]\n", x1, x2);

    // === 3. Define target (correct answer) ===
    // We want: 1 1 → YES (1), others → NO (0)
    double target = (x1 == 1.0 && x2 == 1.0) ? 1.0 : 0.0;
    printf("Target output: %.0f\n", target);

    // === 4. Initialize weights and bias (randomly or fixed) ===
    double w1 = 0.3;   // Weight for input 1
    double w2 = 0.4;   // Weight for input 2
    double b  = -0.5;  // Bias
    double learning_rate = 0.5;  // How much to adjust per step

    printf("\nInitial weights: w1=%.3f, w2=%.3f, bias=%.3f\n", w1, w2, b);

    // === 5. Forward pass BEFORE training ===
    double z = w1 * x1 + w2 * x2 + b;        // Linear combination
    double prediction = relu(z);            // Apply ReLU
    printf("\nBEFORE training: z = %.3f → output = %.3f\n", z, prediction);

    // === 6. TRAINING LOOP (10 epochs) ===
    printf("\n--- Training Start (Backpropagation) ---\n");
    for (int epoch = 1; epoch <= 10; epoch++) {
        // ---- Forward Pass ----
        z = w1 * inputs[0] + w2 * inputs[1] + b;
        double output = relu(z);

        // ---- Compute Error ----
        double error = target - output;  // How wrong are we?

        // ---- If ReLU is off (z <= 0), no learning! ----
        if (z <= 0 && target > 0) {
            printf("Epoch %2d: DEAD NEURON (z=%.3f <=0), no update!\n", epoch, z);
            continue;
        }

        // ---- Backpropagation: Compute gradients ----
        double d_output = error;                    // dL/d_output
        double d_z = d_output * relu_derivative(z); // dL/dz = dL/d_output * d_output/dz

        // Gradients for each parameter
        double d_w1 = d_z * inputs[0];   // dL/dw1
        double d_w2 = d_z * inputs[1];   // dL/dw2
        double d_b  = d_z;               // dL/db

        // ---- Update weights and bias (Gradient Descent) ----
        w1 += learning_rate * d_w1;
        w2 += learning_rate * d_w2;
        b  += learning_rate * d_b;

        // ---- Print progress ----
        printf("Epoch %2d: output=%.3f, error=%.3f → w1=%.3f, w2=%.3f, b=%.3f\n",
               epoch, output, error, w1, w2, b);
    }

    // === 7. Final Forward Pass AFTER training ===
    z = w1 * inputs[0] + w2 * inputs[1] + b;
    double final_output = relu(z);
    printf("\n--- Training Complete ---\n");
    printf("Final z = %.3f → Final output = %.3f\n", z, final_output);
    printf("Final decision: %s\n", (final_output > 0) ? "YES" : "NO");

    return 0;
}
