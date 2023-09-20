#include <cuda.h>
#include <stdio.h>

// Define a struct to represent a point in 3D space.
struct Point3D {
  float x;
  float y;
  float z;
};

// Define a kernel function to calculate the distance between two points.
__global__ void distance(Point3D *p1, Point3D *p2, float *distance) {
  // Calculate the distance between the two points.
  float dx = p1->x - p2->x;
  float dy = p1->y - p2->y;
  float dz = p1->z - p2->z;

  // Square the distances and add them together.
  float distance_squared = dx * dx + dy * dy + dz * dz;

  // Take the square root of the distance squared to get the distance.
  *distance = sqrtf(distance_squared);
}

int main() {
  // Create two Point3D structs on the host.
  Point3D p1 = {9.0f, 2.0f, 3.0f};
  Point3D p2 = {10.0f, 5.0f, 6.0f};

  // Allocate memory for the Point3D structs on the device.
  Point3D *p1_device;
  Point3D *p2_device;
  cudaMalloc(&p1_device, sizeof(Point3D));
  cudaMalloc(&p2_device, sizeof(Point3D));

  // Copy the Point3D structs from the host to the device.
  cudaMemcpy(p1_device, &p1, sizeof(Point3D), cudaMemcpyHostToDevice);
  cudaMemcpy(p2_device, &p2, sizeof(Point3D), cudaMemcpyHostToDevice);

  // Allocate memory for the distance on the device.
  float *distance_device;
  cudaMalloc(&distance_device, sizeof(float));

  // Call the kernel function to calculate the distance between the two points.
  distance<<<1, 1>>>(p1_device, p2_device, distance_device);

  // Copy the distance from the device to the host.
  float distance;
  cudaMemcpy(&distance, distance_device, sizeof(float), cudaMemcpyDeviceToHost);

  // Free the memory on the device.
  cudaFree(p1_device);
  cudaFree(p2_device);
  cudaFree(distance_device);

  // Print the distance between the two points.
  printf("The distance between the two points is %f.\n", distance);

  return 0;
}