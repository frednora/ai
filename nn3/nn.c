

/*
# Compile
gcc -o nn nn.c -lm

# Test all combinations
./nn 1 1 1    # Should → OPENED!
./nn 1 1 0    # Should → CLOSED
./nn 1 0 1    # Should → CLOSED
./nn 0 1 1    # Should → CLOSED
./nn 0 0 0    # Should → CLOSED
*/

/*
Real-World Uses of 3-Input AND Logic(Only when ALL THREE are true → action happens)
#Scenario                   Input 1             Input 2              Input 3               Output (Action)
1 Bank Vault Security       Keycard inserted    Fingerprint matched  Voice code correct    Vault opens
2 Car Engine Start          Key in ignition     Brake pressed        Gear in Park          Engine starts
3 Medical Alert System      Heart rate too low  Blood oxygen low     No movement detected  Call 911
4 Factory Safety Gate       Sensor A: no object Sensor B: no object  Sensor C: no object   Gate opens
5 Login with 3-Factor Auth  Password correct    SMS code entered     Biometric scan passed Login allowed
6 Drone Takeoff Checklist   GPS locked          Battery > 30%        No-fly zone clear     Takeoff approved
7 Chemical Reactor Start    Temp in range       Pressure stable      Safety valve closed   Start reaction
8 Smart Home Lights ON      Motion detected     It's dark            User is home          Turn on lights
9 Credit Card Approval      Income verified     Credit score > 700   No recent defaults    Approve card
10 Nuclear Launch (Fiction) President key       General key          AI confirmation       Launch missiles

*/


// =======================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// -----------------------------
// ReLU Activation
// -----------------------------
double relu(double z) {
    return (z > 0) ? z : 0.0;  // Output z if positive, else 0
}

// -----------------------------
// ReLU Derivative (for backprop)
// -----------------------------
double relu_derivative(double z) {
    return (z > 0) ? 1.0 : 0.0;  // Slope: 1 if z > 0, else 0
}

// -----------------------------
// Main Function
// -----------------------------
int main(int argc, char *argv[]) {
    // === 1. Check input: need 4 args (program + 3 inputs) ===
    if (argc != 4) {
        printf("Usage: %s <key1> <key2> <key3>\n", argv[0]);
        printf("Example: %s 1 1 1\n", argv[0]);
        return 1;
    }

    // === 2. Read 3 inputs ===
    double x1 = atof(argv[1]);
    double x2 = atof(argv[2]);
    double x3 = atof(argv[3]);
    double inputs[3] = {x1, x2, x3};

    printf("=== 3-Key Safe Brain (AND Logic) ===\n");
    printf("Keys: [%.0f, %.0f, %.0f]\n", x1, x2, x3);

    // === 3. Target: only [1,1,1] → YES (1), others → NO (0) ===
    double target = (x1 == 1.0 && x2 == 1.0 && x3 == 1.0) ? 1.0 : 0.0;
    printf("Target: %.0f (1 = all keys in)\n", target);

    // === 4. Initialize weights and bias ===
    double w1 = 0.2;   // Weight for key 1
    double w2 = 0.3;   // Weight for key 2
    double w3 = 0.4;   // Weight for key 3
    double b  = -0.8;  // Bias (threshold)
    double learning_rate = 0.5;

    printf("\nInitial: w1=%.3f, w2=%.3f, w3=%.3f, b=%.3f\n", w1, w2, w3, b);

    // === 5. Forward pass BEFORE training ===
    double z = w1*x1 + w2*x2 + w3*x3 + b;
    double output = relu(z);
    printf("\nBEFORE: z = %.3f → output = %.3f\n", z, output);

    // === 6. TRAINING: 15 epochs ===
    printf("\n--- Training (Backpropagation) ---\n");
    for (int epoch = 1; epoch <= 15; epoch++) {
        // ---- Forward ----
        z = w1*inputs[0] + w2*inputs[1] + w3*inputs[2] + b;
        output = relu(z);

        // ---- Error ----
        double error = target - output;

        // ---- Skip if neuron is dead (z <= 0 and we need output > 0) ----
        if (z <= 0 && target > 0) {
            printf("Epoch %2d: DEAD (z=%.3f), no update\n", epoch, z);
            continue;
        }

        // ---- Backpropagation: gradients ----
        double d_z = error * relu_derivative(z);  // dL/dz

        double d_w1 = d_z * inputs[0];
        double d_w2 = d_z * inputs[1];
        double d_w3 = d_z * inputs[2];
        double d_b  = d_z;

        // ---- Update weights ----
        w1 += learning_rate * d_w1;
        w2 += learning_rate * d_w2;
        w3 += learning_rate * d_w3;
        b  += learning_rate * d_b;

        // ---- Print ----
        printf("Epoch %2d: out=%.3f, err=%.3f → w1=%.3f, w2=%.3f, w3=%.3f, b=%.3f\n",
               epoch, output, error, w1, w2, w3, b);
    }

    // === 7. Final prediction ===
    z = w1*inputs[0] + w2*inputs[1] + w3*inputs[2] + b;
    double final_output = relu(z);
    printf("\n--- DONE ---\n");
    printf("Final z = %.3f → output = %.3f\n", z, final_output);
    printf("SAFE: %s\n", (final_output > 0) ? "OPENED!" : "CLOSED");

    return 0;
}