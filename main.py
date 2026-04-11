import numpy as np
import scipy.stats as st
import heapq

## M/M/1 QUEUE SIMULATOR
def run_simulation(lambda_rate, mu_rate, max_time):
    # queue of events as a min-heap
    event_list = []
    heapq.heapify(event_list)

    # initial conditions and system state
    current_time = 0
    server_busy = False
    queue_length = 0
    first_arrival_time = np.random.exponential(1/lambda_rate)
    heapq.heappush(event_list, (first_arrival_time, 'arrival'))  # first arrival
    print(f"First arrival scheduled at time {first_arrival_time:.4f}")
    heapq.heappush(event_list, (max_time, 'end'))   # end of simulation event
    print(f"End of simulation scheduled at time {max_time:.4f}")
    
    # run until we get to the end of simulation event
    while True:
        (current_time, event_type) = heapq.heappop(event_list)
        print(f"Processing event at time {current_time:.4f}: {event_type}")
        if event_type == 'arrival':
            # schedule the next arrival
            next_arrival_time = current_time + np.random.exponential(1/lambda_rate)
            if next_arrival_time < max_time:
                heapq.heappush(event_list, (next_arrival_time, 'arrival'))
                print(f"Next arrival scheduled at time {next_arrival_time:.4f}")
            if not server_busy:
                server_busy = True
                print("Server is now busy.")
                # schedule the departure of this event
                departure_time = current_time + np.random.exponential(1/mu_rate)
                if departure_time < max_time:
                    heapq.heappush(event_list, (departure_time, 'departure'))
                    print(f"Departure scheduled at time {departure_time:.4f}")
            else:
                # server is busy, event enqued
                queue_length += 1
                print(f"Event queued. Current queue length: {queue_length}")
            
        elif event_type == 'departure':
            # TODO: track the arrival times of events to calculate delays (current_time - its_arrival_times)
            print(f"Event departed at time {current_time:.4f}")
            if queue_length > 0:
                # seize the server with the next event in the queue
                queue_length -= 1
                print(f"Event served from queue. Current queue length: {queue_length}")
                # schedule the departure of this event
                departure_time = current_time + np.random.exponential(1/mu_rate)
                if departure_time < max_time:
                    heapq.heappush(event_list, (departure_time, 'departure'))
                    print(f"Departure scheduled at time {departure_time:.4f}")
            else:
                # no events in queue, server becomes idle
                server_busy = False
                print("Server is now idle.")
        elif event_type == 'end':
            print(f"End of simulation reached at time {current_time:.4f}")
            break

## MAIN
if __name__ == "__main__":
    lambda_rate = 1
    mu_rate = 2
    max_time = 10
    
    run_simulation(lambda_rate, mu_rate, max_time)

    print("Simulation completed.")