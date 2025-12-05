#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <omp.h>
#include <cstdlib> 
#include <tuple>

#define G 6.67430e-8     // Gravitational constant
#define DT 0.1           // Time step
#define NUM_BODIES 4096  // Number of bodies
#define NUM_STEPS 8000   // Number of simulation steps
#define WIDTH 1000.0     // Visualization width
#define HEIGHT 1000.0    // Visualization height
#define DEPTH 1000.0     // Visualization depth

struct Body {
    double x, y, z, vx, vy, vz, mass;
};

std::vector<Body> bodies(NUM_BODIES);
std::vector<std::tuple<double, double, double>> forces(NUM_BODIES);

void compute_forces() {
    for (int i = 0; i < NUM_BODIES; i++) {
        double fx = 0, fy = 0, fz = 0;

        for (int j = 0; j < NUM_BODIES; j++) {
            if (i == j) continue;

            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dz = bodies[j].z - bodies[i].z;

            double dist = sqrt(dx*dx + dy*dy + dz*dz) + 1e-9;

            double F = (G * bodies[i].mass * bodies[j].mass) / (dist * dist);

            fx += F * dx / dist;
            fy += F * dy / dist;
            fz += F * dz / dist;
        }

        forces[i] = {fx, fy, fz};
    }
}


void update_positions() {
    for (int i = 0; i < NUM_BODIES; i++) {
        double fx = std::get<0>(forces[i]);
        double fy = std::get<1>(forces[i]);
        double fz = std::get<2>(forces[i]);

        bodies[i].vx += (fx / bodies[i].mass) * DT;
        bodies[i].vy += (fy / bodies[i].mass) * DT;
        bodies[i].vz += (fz / bodies[i].mass) * DT;

        bodies[i].x += bodies[i].vx * DT;
        bodies[i].y += bodies[i].vy * DT;
        bodies[i].z += bodies[i].vz * DT;

        // Boundary reflections
        if (bodies[i].x < 0 || bodies[i].x > WIDTH) bodies[i].vx *= -1;
        if (bodies[i].y < 0 || bodies[i].y > HEIGHT) bodies[i].vy *= -1;
        if (bodies[i].z < 0 || bodies[i].z > DEPTH) bodies[i].vz *= -1;
    }
}


void initialize_bodies() {
    for (int i = 0; i < NUM_BODIES; i++) {
        bodies[i] = {
            (double)(rand() % 1000),
            (double)(rand() % 1000),
            (double)(rand() % 1000),
            0.0, 0.0, 0.0,
            1e10
        };
    }
}


void save_to_csv(std::ofstream &file, int step) {
    for (int i = 0; i < NUM_BODIES; i++) {
        file << step << "," << i << ","
             << bodies[i].x << "," << bodies[i].y << "," << bodies[i].z << ","
             << bodies[i].vx << "," << bodies[i].vy << "," << bodies[i].vz
             << "\n";
    }
}

void run_simulation() {
    double start = omp_get_wtime();
    std::ofstream file("nbody_output.csv");
    file << "step,id,x,y,z\n";

    initialize_bodies();

    for (int step = 0; step < NUM_STEPS; step++) {
        compute_forces();
        update_positions();

        // save positions for visualization
        if (step % 10 == 0)  // reduce file size
            save_to_csv(file, step);

        if (step % 500 == 0)
            std::cout << "Completed step " << step << "/" << NUM_STEPS << "\n";
    }

    file.close();
double end = omp_get_wtime();
std::cout << "Runtime: " << (end - start) << " seconds\n";
}


int main(int argc, char *argv[]) {
    run_simulation();
    return 0;
}

