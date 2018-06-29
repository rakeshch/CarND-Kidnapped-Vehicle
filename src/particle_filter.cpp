/*
 * particle_filter.cpp
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

using Eigen::MatrixXd;
using Eigen::VectorXd;

#define NUMBER_OF_PARTICLES 500 // Change this value to test with different number of particles
#define EPS 0.0001

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if (!is_initialized)
	{
		//initialize number of particles to use
		num_particles = NUMBER_OF_PARTICLES;

		random_device rd;
		// Define random generator with Gaussian distribution
		default_random_engine gen(rd());

		// Create normal (Gaussian) distribution for x,y and theta.
		normal_distribution<double> dist_x(x, std[0]);
		normal_distribution<double> dist_y(y, std[1]);
		normal_distribution<double> dist_theta(theta, std[2]);

		// Resize the particles vector to fit desired number of particles
		particles.resize(num_particles);
		weights.resize(num_particles);

		//initialize all particles
		for (int i = 0; i < num_particles; i++)
		{
			particles[i].id = i;
			particles[i].x = dist_x(gen);
			particles[i].y = dist_y(gen);
			particles[i].theta = dist_theta(gen);
			particles[i].weight = 1.0;
		}
		is_initialized = true;
		return;
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	random_device rd;
	// Define random generator with Gaussian distribution
	default_random_engine gen(rd());

	// Create normal (Gaussian) distribution for x,y and theta.
	normal_distribution<double> dist_x(0., std_pos[0]);
	normal_distribution<double> dist_y(0., std_pos[1]);
	normal_distribution<double> dist_theta(0., std_pos[2]);

	//constant to use in iterations below	
	const double vel_delta = velocity * delta_t;

	for (int i = 0; i < num_particles; i++)
	{
		if (fabs(yaw_rate) > EPS)
		{
			double vel_yaw_ratio = velocity / yaw_rate;
			double yaw_delta = yaw_rate * delta_t;

			// also adding noise dist_x(gen) to x,y and theta
			particles[i].x += vel_yaw_ratio * (sin(particles[i].theta + yaw_delta) - sin(particles[i].theta)) + dist_x(gen);
			particles[i].y += vel_yaw_ratio * (cos(particles[i].theta) - cos(particles[i].theta + yaw_delta)) + dist_y(gen);
			particles[i].theta += yaw_delta + dist_theta(gen);
		}
		else
		{
			// also adding noise dist_x(gen) to x and y (theta is zero)
			particles[i].x += vel_delta * cos(particles[i].theta) + dist_x(gen);
			particles[i].y += vel_delta * sin(particles[i].theta) + dist_y(gen);
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	//constants to use in weight calculations
	const double sigma_x2 = std_landmark[0] * std_landmark[0];
	const double sigma_y2 = std_landmark[1] * std_landmark[1];
	const double f = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

	// save sum of weights for normalization
	double sum_w = 0.0;

	unsigned int nObservations = observations.size();

	for (int i = 0; i < num_particles; ++i)
	{
		// Iterate through all the observations and find the shortest distance between each observation and the particle
		for (unsigned int j = 0; j < nObservations; ++j)
		{
			// 1. Transform each observation marker from the vehicle coordinates to map coordinates, with respect to the particle
			LandmarkObs observation;
			observation.id = observations[j].id;
			observation.x = particles[i].x + (observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta));
			observation.y = particles[i].y + (observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta));

			// 2. For available map landmarks, find the nearest landmark to the transformed observation
			Map::single_landmark_s near_landmark;
			bool in_sensor_range = false;
			double short_dist = numeric_limits<double>::max();

			for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); ++k)
			{
				// calculate Euclidean distance between transformed observation and the landmark
				double distance = dist(map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f, observation.x, observation.y);
				if (distance < short_dist)
				{
					//update the value of short_dist
					short_dist = distance;

					//assign this landmark as the nearest one to the observation
					near_landmark = map_landmarks.landmark_list[k];

					//check if the distance is within sensor range
					if (distance < sensor_range)
					{
						in_sensor_range = true;
					}
				}
			}

			// 3. If the landmark is in given sensor range, calculate the weight of the particle with respect to in_range observation and nearest landmark
			if (in_sensor_range == true)
			{
				//calculate weight
				double dx = observation.x - near_landmark.x_f;
				double dy = observation.y - near_landmark.y_f;

				// calculate multivariable-gaussian (weight)
				double weight = f * exp(-0.5 * ((dx * dx / sigma_x2) + (dy * dy / sigma_y2)));

				// final weight of the particle will be the product of each measurement's multivariable-gaussian probability density (weight)
				particles[i].weight *= weight;
			}
			else
			{
				particles[i].weight *= EPS;
			}	
			//sum of weights to be used later, for normalization
			sum_w += particles[i].weight;
		}
	}
	// Weights normalization to sum of weights=1
	for (int l = 0; l < num_particles; ++l) {
		particles[l].weight /= sum_w;
		weights[l] = particles[l].weight;
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	random_device rd;
	static default_random_engine gen(rd());

	discrete_distribution<> dis_particles(weights.begin(), weights.end());
	vector<Particle> new_particles;
	new_particles.resize(num_particles);
	for (int i = 0; i < num_particles; i++) {
		new_particles[i] = particles[dis_particles(gen)];
	}
	particles = move(new_particles);
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
