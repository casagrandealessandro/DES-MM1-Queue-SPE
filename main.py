import numpy as np
import scipy.stats as st
import heapq
import collections
import matplotlib.pyplot as plt

## M/M/1 QUEUE SIMULATOR
def run_simulation(lambda_rate, mu_rate, max_time, ex, rng):

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
                # schedule the departure of this event, according to the chosen service time distribution
                if ex == 1:
                    departure_time = current_time + rng.exponential(1/mu_rate)
                elif ex == 2:
                    departure_time = current_time + rejection_sampling(rng)
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
                # schedule the departure of this event, according to the chosen service time distribution
                if ex == 1:
                    departure_time = current_time + rng.exponential(1/mu_rate)
                elif ex == 2:
                    departure_time = current_time + rejection_sampling(rng)
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

# INDEPENDENT REPLICATIONS + PLOTS
# runs are independent because they use different RNG seeds (=> independent)
# each run has same initial conditions (=> identically distributed)
def run_scenario(lambda_rate, mu_rate, max_time, n_runs, ex, master_seed):
    average_delays_per_run = []
    all_running_avg_delays = []
    all_departure_times = []

    print("Running simulations...")

    for i in range(n_runs):
        rng = np.random.default_rng(master_seed + i)
        delays, running_avg_delays, departure_times = run_simulation(lambda_rate, mu_rate, max_time, ex, rng)
        average_delays_per_run.append(np.mean(delays))
        all_running_avg_delays.append(running_avg_delays)
        all_departure_times.append(departure_times)

    print("Simulations completed")

    # plots: all running average delays over time
    plt.figure()
    for run_avg, dep_times in zip(all_running_avg_delays, all_departure_times):
        plt.plot(dep_times, run_avg, alpha=0.3, linewidth=0.5)
    if mu_rate is not None:
        plt.axhline(1/(mu_rate-lambda_rate), color='black', linestyle='--', label='Theoretical Average', linewidth=1.5)
    plt.xlim(0, max_time)
    plt.xlabel('Time [s]', fontsize=13)
    plt.ylabel('Average delay in System [s]', fontsize=13)
    plt.title(f'Average delay in system vs. Time (λ={lambda_rate}, μ={mu_rate})', fontsize=15)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    if mu_rate is not None:
        plt.legend(fontsize=11)
    filename = f"Images/plot_lambda{lambda_rate}_mu{mu_rate}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()

    # Q-Q plot to check for normality of the average delays
    plt.figure()
    st.probplot(average_delays_per_run, dist="norm", plot=plt)
    plt.title(f'Q-Q plot for average delays (λ={lambda_rate}, μ={mu_rate})', fontsize=15)
    plt.xlabel('Theoretical Quantiles', fontsize=13)
    plt.ylabel('Ordered Values', fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    qq_filename = f"Images/qq_plot_lambda{lambda_rate}_mu{mu_rate}.pdf"
    plt.savefig(qq_filename, format="pdf", bbox_inches="tight")
    plt.show()

    print("Statistics:")

    # in order to correctly apply the confidence interval formula, these assumptions must hold:
    # 1. The average delays from each run are independent and identically distributed (true by IR method design)
    # 2. The average delays from each run are approximately normally distributed (holds for large n_runs by Central Limit Theorem)
    # 3. The duration of the simulation is long enough to ensure the transient is negligible and the system is in steady state (TODO: think about cutting out the transient, or applying batch means?)
    grand_mean = np.mean(average_delays_per_run)
    confidence_interval = st.t.interval(0.95, df=n_runs-1, loc=grand_mean, scale=st.sem(average_delays_per_run))
    if mu_rate is not None:
        print(f"Theoretical Average: {1 / (mu_rate - lambda_rate):.3f}s")
    print(f"Simulated Average: {grand_mean:.3f}s")
    print(f"95% Confidence Interval: [{confidence_interval[0]:.3f}s, {confidence_interval[1]:.3f}s]")

# REJECTION SAMPLING
# to draw from the distribution f(x) = A|sinc(x-3)| for x in [0,6].
# the sinc function has a maximum in x=3, let A=1 => the maximum is 1 and the bounding box is in [0,6] with height 1
def rejection_sampling(rng):
    while True:
        x = rng.uniform(0, 6)
        y = rng.uniform(0, 1)
        if y < abs(np.sinc(x - 3)):
            return x

## MAIN
if __name__ == "__main__":
    n_runs = 30
    initial_seed = 666

    print("\nEXERCISE 1")

    # default scenario: λ=1, μ=2
    lambda_rate = 1
    mu_rate = 2
    max_time = 30000
    print(f"Default scenario: λ={lambda_rate}, μ={mu_rate}")
    run_scenario(lambda_rate, mu_rate, max_time, n_runs, 1, initial_seed)

    # high load scenario: λ=1.9, μ=2
    lambda_rate = 1.9
    mu_rate = 2
    max_time = 400000  # to ensure steady state is reached
    print(f"\nHigh load scenario: λ={lambda_rate}, μ={mu_rate}")
    run_scenario(lambda_rate, mu_rate, max_time, n_runs, 1, initial_seed)

    # low load scenario: λ=1, μ=10
    lambda_rate = 1
    mu_rate = 10
    max_time = 8000  # very stable
    print(f"\nLow load scenario: λ={lambda_rate}, μ={mu_rate}")
    run_scenario(lambda_rate, mu_rate, max_time, n_runs, 1, initial_seed)

    print("\nEXERCISE 2")

    # plot the shape of the service time distribution
    x = np.linspace(0, 6, 1000)
    plt.figure()
    plt.plot(x, np.abs(np.sinc(x - 3)), linewidth=2)
    plt.xlim(0, 6)
    plt.ylim(0, 1.1)
    plt.xlabel('Service Time', fontsize=13)
    plt.ylabel('Density f(x)=|sinc(x-3)|', fontsize=13)
    plt.title('Service Time Distribution', fontsize=15)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig("Images/service_time_distribution.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # for the system to be stable: λ < 1/μ. μ=3 => λ < 1/3 ≈ 0.333
    # stable scenario: λ=0.3
    max_time = 100000
    lambda_rate = 0.3
    print(f"\nStable scenario: λ={lambda_rate}")
    run_scenario(lambda_rate, None, max_time, n_runs, 2, initial_seed)

    # extreme scenario: λ=0.33 (almost equal to the stability limit)
    max_time = 1000000
    lambda_rate = 0.33
    print(f"\nExtreme scenario: λ={lambda_rate}")
    run_scenario(lambda_rate, None, max_time, n_runs, 2, initial_seed)