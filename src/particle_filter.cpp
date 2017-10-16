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
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;

    std::default_random_engine gen;

    std::normal_distribution<double> N_x(x, std[0]);
    std::normal_distribution<double> N_y(y, std[1]);
    std::normal_distribution<double> N_theta(theta, std[3]);

    for (int i = 0; i < num_particles; ++i) {
        Particle new_particle;
        new_particle.id = i;
        new_particle.x = N_x(gen);
        new_particle.y = N_y(gen);
        new_particle.theta = N_theta(gen);
        new_particle.weight = 1;

        particles.push_back(new_particle);
        weights.push_back(1);

    }
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    std::default_random_engine gen;

    for (int i = 0; i < num_particles; i++) {

        double new_x;
        double new_y;
        double new_theta;

        if (yaw_rate == 0) {
            new_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
            new_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
            new_theta = particles[i].theta;
        } else {
            new_x = particles[i].x +
                    velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            new_y = particles[i].y +
                    velocity / yaw_rate * (cos(particles[i].theta - cos(particles[i].theta + delta_t * yaw_rate)));
            new_theta = particles[i].theta + yaw_rate * delta_t;
        }

        std::normal_distribution<double> N_x(new_x, std_pos[0]);
        std::normal_distribution<double> N_y(new_y, std_pos[1]);
        std::normal_distribution<double> N_theta(new_theta, std_pos[3]);


        particles[i].x = N_x(gen);
        particles[i].y = N_y(gen);
        particles[i].theta = N_theta(gen);

    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.


    for (int i = 0; i < observations.size(); i++) {

        // leave loop, if no landmarks in proximity.
        if (predicted.empty()) break;
        // for each observation, run over each landmark close enough to the vehicle

        // initialize closest landmark id and distance placeholders
        int closest_predicted_landmark_id = 0;
        double closest_predicted_landmark_distance = 0;



        for (int k = 0; k < predicted.size(); k++) {
            // init best fit = prediction one, but only if not set
            // calculate the distance from observation and distance. if it is closer than all distances before, make the observation the best fit

            double distance;

            distance = dist(predicted[k].x, predicted[k].y, observations[i].x, observations[i].y);


            if (k == 0){
                closest_predicted_landmark_id = predicted[k].id;
                closest_predicted_landmark_distance = distance;
            }
            else if (closest_predicted_landmark_distance < distance ){
                closest_predicted_landmark_distance =  distance;
            }


        }
        observations[i].id = closest_predicted_landmark_id;
    }


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // ∆ Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html



    for (int p = 0; p < particles.size(); p++) {


        //1. Make list of all landmarks within sensor range of particle, call this `predicted_lm`

        vector<LandmarkObs> predicted_lm;

        for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {

            double distance;

            distance = dist(map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f,
                            particles[p].x, particles[p].y);

            if (distance <= sensor_range){
                LandmarkObs temp;
                temp.x = map_landmarks.landmark_list[k].x_f;
                temp.y = map_landmarks.landmark_list[k].y_f;
                temp.id = map_landmarks.landmark_list[k].id_i;
                predicted_lm.push_back(temp);
            }
        }


        //2. Convert all observations from local to global frame, call this `transformed_obs`
        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;

        vector<LandmarkObs> transformed_observations;
        LandmarkObs obs;
        for (int i = 0; i < observations.size(); i++) {

            LandmarkObs transformed_obs;
            obs = observations[i];

            // homogenious transformation
            transformed_obs.x = particles[p].x*(obs.x*cos(particles[p].theta)-obs.y*sin(particles[p].theta));
            transformed_obs.y = particles[p].y*(obs.x*cos(particles[p].theta)+obs.y*sin(particles[p].theta));
            transformed_observations.push_back(transformed_obs);

        }
        particles[p].weight = 1;


        //3. Perform `dataAssociation`. This will put the index of the `predicted_lm` nearest to each `transformed_obs` in the `id` field of the `transformed_obs` element.

        dataAssociation(predicted_lm, transformed_observations);

        //4. Loop through all the `transformed_obs`. Use the saved index in the `id` to find the associated landmark and compute the gaussian.

        double gaussians_observations [transformed_observations.size()];

        for (int t = 0; t < transformed_observations.size(); t++) {

            double observed_x = transformed_observations[t].x;
            double observed_y = transformed_observations[t].y;

            int associated_prediction = transformed_observations[t].id;

            double predicted_x = 0.0;
            double predicted_y = 0.0;

            for (unsigned int k = 0; k < predicted_lm.size(); k++) {
                if (predicted_lm[k].id == associated_prediction) {
                    predicted_x = predicted_lm[k].x;
                    predicted_y = predicted_lm[k].y;
                }
            }


            //double multiplier = 1.0/(2*M_PI*0.3*0.3);
            double cov_x = pow(std_landmark[0], 2.0);
            double cov_y = pow(std_landmark[1], 2.0);



            double observation_prob_i =
                    exp(-pow(predicted_x - observed_x, 2.0)
                        /(2.0*cov_x) - pow(predicted_y - observed_y, 2.0)/(2.0*cov_y));
            gaussians_observations[t] = observation_prob_i;
        }


        //5. Multiply all the gaussian values together to get total probability of particle (the weight). (edited)

        for (int g = 0; g < sizeof(gaussians_observations); ++g) {
            particles[p].weight *= gaussians_observations[g];

        }

        weights[p] = particles[p].weight;

    }
}

void ParticleFilter::resample() {
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution


    default_random_engine gen;
    discrete_distribution<int> distribution(weights.begin(), weights.end());

    vector<Particle> resample_particles;

    for (int i = 0; i < num_particles; ++i) {
        resample_particles.push_back(particles[distribution(gen)]);
    }

    particles = resample_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

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
