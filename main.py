import numpy as np
import scipy.stats as st
import heapq
import collections
import matplotlib.pyplot as plt

## M/M/1 QUEUE SIMULATOR
def run_simulation(lambda_rate, mu_rate, max_time):
    rng = np.random.default_rng()

    # queue of events as a min-heap
    event_list = []
    heapq.heapify(event_list)

    arrival_times = {}
    delays = []
    running_avg_delays = []
    departure_times = []
    total_delay_sum = 0
    departed_count = 0
    queue = collections.deque()

    # initial conditions and system state
    current_time = 0
    event_id = 0
    server_busy = False
    first_arrival_time = rng.exponential(1/lambda_rate)
    heapq.heappush(event_list, (first_arrival_time, 'arrival', event_id))
    #print(f"First arrival scheduled at time {first_arrival_time:.4f}")
    heapq.heappush(event_list, (max_time, 'end', -1))
    #print(f"End of simulation scheduled at time {max_time:.4f}")

    next_event_id = event_id + 1

    # run until we get to the end of simulation event
    while True:
        (current_time, event_type, event_id) = heapq.heappop(event_list)
        #print(f"Processing event at time {current_time:.4f}: {event_type}")
        
        if event_type == 'arrival':
            arrival_times[event_id] = current_time
            # schedule the next arrival
            next_arrival_time = current_time + rng.exponential(1/lambda_rate)
            heapq.heappush(event_list, (next_arrival_time, 'arrival', next_event_id))
            #print(f"Next arrival scheduled at time {next_arrival_time:.4f}")
            next_event_id += 1
            if not server_busy:
                server_busy = True
                #print("Server is now busy")
                # schedule the departure of this event
                departure_time = current_time + rng.exponential(1/mu_rate)
                heapq.heappush(event_list, (departure_time, 'departure', event_id))
                #print(f"Departure scheduled at time {departure_time:.4f}")
            else:
                # server is busy, event queued
                queue.append(event_id)
                #print(f"Event queued. Current queue: {list(queue)}")
        elif event_type == 'departure':
            #print(f"Event departed at time {current_time:.4f}")
            # compute the delay for this event
            delay = current_time - arrival_times[event_id]
            delays.append(delay)
            total_delay_sum += delay
            departed_count += 1
            running_avg_delays.append(total_delay_sum / departed_count)
            departure_times.append(current_time)
            if queue:
                # seize the server with the next event in the queue
                next_id = queue.popleft()
                #print(f"Event served from queue (event_id={next_id}). Current queue: {list(queue)}")
                # schedule the departure of this event
                departure_time = current_time + rng.exponential(1/mu_rate)
                heapq.heappush(event_list, (departure_time, 'departure', next_id))
                #print(f"Departure scheduled at time {departure_time:.4f}")
            else:
                # no events in queue, server becomes idle
                server_busy = False
                #print("Server is now idle")
        
        elif event_type == 'end':
            #print(f"End of simulation reached at time {current_time:.4f}")
            break
    return delays, running_avg_delays, departure_times

# independent replications (IR) method is employed:
# runs are independent because they use different RNG seeds, automatically set by system entropy (=> independent)
# each run has same initial conditions (=> identically distributed)
def run_scenario(lambda_rate, mu_rate, max_time, n_runs):
    average_delays_per_run = []
    all_running_avg_delays = []
    all_departure_times = []

    for i in range(n_runs):
        #print(f"\n--- Simulation run {i+1} ---")
        delays, running_avg_delays, departure_times = run_simulation(lambda_rate, mu_rate, max_time)
        average_delays_per_run.append(np.mean(delays))
        all_running_avg_delays.append(running_avg_delays)
        all_departure_times.append(departure_times)

    # plots: all running average delays over time
    plt.figure()
    for run_avg, dep_times in zip(all_running_avg_delays, all_departure_times):
        plt.plot(dep_times, run_avg, alpha=0.3, linewidth=0.7)
    plt.axhline(1/(mu_rate-lambda_rate), color='red', linestyle='--', label='Theoretical Average')
    plt.xlabel('Time [s]')
    plt.ylabel('Average Delay in System [s]')
    plt.title(f'Running Average Delay vs Time for {n_runs} Replications (λ={lambda_rate}, μ={mu_rate})')
    plt.legend()
    filename = f"plot_lam{lambda_rate}_mu{mu_rate}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()

    print("Simulation completed. Computing final statistics...")

    # in order to correctly apply the confidence interval formula, these assumptions must hold:
    # 1. The average delays from each run are independent and identically distributed (true by IR method design)
    # 2. The average delays from each run are approximately normally distributed (holds for large n_runs by Central Limit Theorem, TODO: Q-Q plot to check normality)
    # 3. The duration of the simulation is long enough to ensure the transient is negligible and the system is in steady state (TODO: plot average delay over time to check for steady state)
    grand_mean = np.mean(average_delays_per_run)
    confidence_interval = st.t.interval(0.95, df=n_runs-1, loc=grand_mean, scale=st.sem(average_delays_per_run))
    print(f"Theoretical Average: {1 / (mu_rate - lambda_rate):.3f} s")
    print(f"Simulated Average: {grand_mean:.3f} s")
    print(f"95% Confidence Interval: {confidence_interval[0]:.3f} to {confidence_interval[1]:.3f} s")

## MAIN
if __name__ == "__main__":
    max_time = 30000
    n_runs = 30

    # default scenario: λ=1, μ=2
    lambda_rate = 1
    mu_rate = 2
    print(f"Default scenario: λ={lambda_rate}, μ={mu_rate}")
    run_scenario(lambda_rate, mu_rate, max_time, n_runs)

    # high load scenario: λ=1.9, μ=2
    lambda_rate = 1.9
    mu_rate = 2
    max_time = 400000  # to ensure steady state is reached
    print(f"\nHigh load scenario: λ={lambda_rate}, μ={mu_rate}")
    run_scenario(lambda_rate, mu_rate, max_time, n_runs)

    # low load scenario: λ=1, μ=10
    lambda_rate = 1
    mu_rate = 10
    max_time = 10000  # very stable
    print(f"\nLow load scenario: λ={lambda_rate}, μ={mu_rate}")
    run_scenario(lambda_rate, mu_rate, max_time, n_runs)