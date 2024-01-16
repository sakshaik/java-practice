package com.test.euler;
// https://projecteuler.net/problem=7
public class Problem7 {

	public static void main(String[] args) {
		System.out.println(nthPrime(10001));
	}

	/* returns the nth prime number */
	public static long nthPrime(long n) {
		int numberOfPrimes = 0;
		long prime = 1;

		while (numberOfPrimes < n) {
			prime++;
			if (isPrime(prime)) {
				numberOfPrimes++;
			}
		}
		return prime;
	}

	/*
	 * returns true if parameter n is a prime number, false if composite or neither
	 */
	public static boolean isPrime(long n) {
		if (n < 2)
			return false;
		else if (n == 2)
			return true;
		for (int i = 2; i < Math.sqrt(n) + 1; i++) {
			if (n % i == 0)
				return false;
		}
		return true;
	}

}
