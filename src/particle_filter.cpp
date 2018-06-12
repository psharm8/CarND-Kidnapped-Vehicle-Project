/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;
    normal_distribution<double> norm_x(x, std[0]);
    normal_distribution<double> norm_y(y, std[1]);
    normal_distribution<double> norm_theta(theta, std[2]);

    for (int i = 0; i < num_particles; ++i) {
        Particle p;
        p.id = i;
        p.x = norm_x(gen);
        p.y = norm_y(gen);
        p.theta = norm_theta(gen);
        p.weight = 1.0;

        particles.push_back(p);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    normal_distribution<double> norm_x(0, std_pos[0]);
    normal_distribution<double> norm_y(0, std_pos[1]);
    normal_distribution<double> norm_theta(0, std_pos[2]);

    for (Particle &p: particles) {
        const double cos_theta = cos(p.theta);
        const double sin_theta = sin(p.theta);
        if (fabs(yaw_rate) < 0.00001) {
            p.x += velocity * delta_t * cos_theta;
            p.y += velocity * delta_t * sin_theta;
        } else {
            const double yawdt = yaw_rate * delta_t;
            const double vrate = velocity / yaw_rate;
            p.x += vrate * (sin(p.theta + yawdt) - sin_theta);
            p.y += vrate * (cos_theta - cos(p.theta + yawdt));
            p.theta += yawdt;
        }

        p.x += norm_x(gen);
        p.y += norm_y(gen);
        p.theta += norm_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    for (LandmarkObs &o:observations) {
        double min_dist = numeric_limits<double>::max();

        int map_i = -1;

        for (LandmarkObs &p:predicted) {
            double d = dist(o.x, o.y, p.x, p.y);
            // Find nearest neighbour
            if (d < min_dist) {
                min_dist = d;
                map_i = p.id;
            }
        }
        // Update the landmark id to the nearest predicted landmark.
        o.id = map_i;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    const double std_x = std_landmark[0];
    const double std_y = std_landmark[1];
    const double gauss_norm = (1 / (2 * M_PI * std_x * std_y));
    for (Particle &p: particles) {
        // Landmarks within sensor range
        vector<LandmarkObs> predictions;
        for (const Map::single_landmark_s &lm:map_landmarks.landmark_list) {
            double radius = dist(p.x, p.y, lm.x_f, lm.y_f);
            if (fabs(radius) <= sensor_range) {
                predictions.emplace_back(lm.id_i, lm.x_f, lm.y_f);
            }
        }


        // Transform Obs to Map coordinates
        const double cos_theta = cos(p.theta);
        const double sin_theta = sin(p.theta);

        vector<LandmarkObs> transformed;
        for (const LandmarkObs &o:observations) {
            double tx = cos_theta * o.x - sin_theta * o.y + p.x;
            double ty = sin_theta * o.x + cos_theta * o.y + p.y;
            transformed.emplace_back(o.id, tx, ty);
        }

        dataAssociation(predictions, transformed);

        p.weight = 1.0;

        for (LandmarkObs &to:transformed) {
            // Find the associated prediction
            double px = 0, py = 0;
            for (LandmarkObs &al:predictions) {
                if (al.id == to.id) {
                    px = al.x;
                    py = al.y;
                    break;
                }
            }
            double exponent = (pow(px - to.x, 2) / (2 * pow(std_x, 2)) + (pow(py - to.y, 2) / (2 * pow(std_y, 2))));
            double obs_weight = gauss_norm*exp(-exponent);
            p.weight *= obs_weight;
        }
    }


}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    vector<Particle> new_particles;

    // Re-sampling wheel

    vector<double> weights;
    for (Particle &p: particles) {
        weights.push_back(p.weight);
    }

    uniform_int_distribution<int> dist_i(0, num_particles-1);
    auto index = dist_i(gen);

    double max_weight = *max_element(weights.begin(), weights.end());

    uniform_real_distribution<double> dist_weight(0.0, max_weight);

    double beta = 0.0;

    for (int i = 0; i < num_particles; i++) {
        beta += dist_weight(gen) * 2.0;
        while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }

    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
                                         const std::vector<double> &sense_x, const std::vector<double> &sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
