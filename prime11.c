#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <gmp.h>
#include <time.h>
#include <pthread.h>
#include <semaphore.h>

#ifndef MAX_THREADS
#define MAX_THREADS 8
#endif

void print_timestamp(void) {
	char buf[32];
	time_t now;
	time(&now);
	strftime(buf, sizeof buf, "%Y/%m/%d %H:%M: ", gmtime(&now));
	printf("%s", buf);
}

#define LOG(...)	{			\
	print_timestamp();			\
	printf(" " __VA_ARGS__);	\
	fflush(stdout);				\
}

// Structure to pass parameters to threads
typedef struct {
	unsigned long p;
} ThreadData;

// Queue structure for tasks
typedef struct {
	ThreadData* tasks;
	int max_tasks;
	int head;
	int tail;
	int count;
	pthread_mutex_t mutex;
	sem_t sem_empty;
	sem_t sem_full;
} TaskQueue;

void init_queue(TaskQueue* queue, int max_tasks) {
	queue->tasks = malloc(sizeof(ThreadData) * max_tasks);
	queue->max_tasks = max_tasks;
	queue->head = 0;
	queue->tail = 0;
	queue->count = 0;
	pthread_mutex_init(&queue->mutex, NULL);
	sem_init(&queue->sem_empty, 0, max_tasks);
	sem_init(&queue->sem_full, 0, 0);
}

void enqueue(TaskQueue* queue, ThreadData data) {
	sem_wait(&queue->sem_empty);
	pthread_mutex_lock(&queue->mutex);
	queue->tasks[queue->tail] = data;
	queue->tail = (queue->tail + 1) % queue->max_tasks;
	queue->count++;
	pthread_mutex_unlock(&queue->mutex);
	sem_post(&queue->sem_full);
}

ThreadData dequeue(TaskQueue* queue) {
	sem_wait(&queue->sem_full);
	pthread_mutex_lock(&queue->mutex);
	ThreadData data = queue->tasks[queue->head];
	queue->head = (queue->head + 1) % queue->max_tasks;
	queue->count--;
	pthread_mutex_unlock(&queue->mutex);
	sem_post(&queue->sem_empty);
	return data;
}

// Lucas-Lehmer Test function
int lucas_lehmer(unsigned long p) {
	mpz_t V, mp, t;
	unsigned long k;
	int res;

	if (p == 2) return 1;  // Special case for Mersenne Prime 3

	// Check if p is prime
	mpz_init_set_ui(t, p);
	if (!mpz_probab_prime_p(t, 25)) {
		mpz_clear(t);
		return 0;
	}

	mpz_init(mp);
	mpz_setbit(mp, p);
	mpz_sub_ui(mp, mp, 1);

	// Special case: if p = 3 mod 4 and both p and 2p+1 are prime, 2p+1 divides 2^p-1
	if (p > 3 && p % 4 == 3) {
		mpz_mul_ui(t, t, 2);
		mpz_add_ui(t, t, 1);
		if (mpz_probab_prime_p(t, 25) && mpz_divisible_p(mp, t)) {
			mpz_clear(mp);
			mpz_clear(t);
			return 0;
		}
	}

	// Small trial division for fast elimination
	unsigned long tlim = p / 2 > (ULONG_MAX / (2 * p)) ? ULONG_MAX / (2 * p) : p / 2;
	for (k = 1; k < tlim; k++) {
		unsigned long q = 2 * p * k + 1;
		if ((q % 8 == 1 || q % 8 == 7) && q % 3 && q % 5 && q % 7 && mpz_divisible_ui_p(mp, q)) {
			mpz_clear(mp);
			mpz_clear(t);
			return 0;
		}
	}

	mpz_clear(t);
	LOG("Lucas-Lehmer is required for M%lu\n", p);

	// Lucas-Lehmer sequence initialization
	mpz_init_set_ui(V, 4);
	for (k = 3; k <= p; k++) {
		mpz_mul(V, V, V);
		mpz_sub_ui(V, V, 2);
		mpz_tdiv_r_2exp(t, V, p);
		mpz_tdiv_q_2exp(V, V, p);
		mpz_add(V, V, t);
		while (mpz_cmp(V, mp) >= 0) mpz_sub(V, V, mp);
	}
	res = !mpz_sgn(V);
	mpz_clear(mp);
	mpz_clear(V);
	return res;
}

// Thread function for finding Mersenne Primes
void* worker_thread(void* arg) {
	TaskQueue* queue = (TaskQueue*)arg;
	while (1) {
		ThreadData data = dequeue(queue);
		if (lucas_lehmer(data.p)) {
			LOG("Discovered Mersenne Prime!! M%lu\n", data.p);
			LOG("Remember to do a full candidacy check.\n");
		} else {
			LOG("-- %lu is not prime.\n", data.p);
		}
	}
	return NULL;
}

int main(int argc, char* argv[]) {
	unsigned long start = (argc >= 2) ? strtoul(argv[1], 0, 10) : 1;
	pthread_t threads[MAX_THREADS];

	TaskQueue queue;
	init_queue(&queue, 100);

	for (int i = 0; i < MAX_THREADS; i++) {
		pthread_create(&threads[i], NULL, worker_thread, &queue);
	}

	for (unsigned long i = start;; i++) {
		ThreadData data = { .p = i };
		enqueue(&queue, data);
	}

	// Unreachable
	for (int i = 0; i < MAX_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}
	free(queue.tasks);
	return 0;
}
