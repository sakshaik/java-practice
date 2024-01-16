package com.test.streams;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class StreamsTest {

	public static void main(String[] args) {
		List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
		System.out.println(StreamsTest.sumOfAllOddNumbersSquare(numbers));
		System.out.println(StreamsTest.sumOfAllEvenNumbersSquare(numbers));
		System.out.println(StreamsTest.sumOfAllEvenNumbersSquareInParallel(numbers));
		System.out.println(StreamsTest.filterAllOddNumbers(numbers));
		System.out.println(StreamsTest.filterAllEvenNumbers(numbers));
		System.out.println(StreamsTest.filterAllNumbersLessThanX(numbers, 3));

		List<List<Integer>> list = Arrays.asList(Arrays.asList(-9, -9, -9, 1, 1, 1), Arrays.asList(0, -9, 0, 4, 3, 2),
				Arrays.asList(-9, -9, -9, 1, 2, 3), Arrays.asList(0, 0, 8, 6, 6, 0), Arrays.asList(0, 0, 0, -2, 0, 0),
				Arrays.asList(0, 0, 1, 2, 4, 0));
		System.out.println(list);
		System.out.println(StreamsTest.convertFromIntToString(numbers));
	}

	private static int sumOfAllOddNumbersSquare(List<Integer> numbers) {
		return numbers.stream().filter(x -> (x % 2 != 0)).mapToInt(x -> x * x).sum();
	}

	private static int sumOfAllEvenNumbersSquare(List<Integer> numbers) {
		return numbers.stream().filter(x -> (x % 2 == 0)).mapToInt(x -> x * x).sum();
	}

	private static List<Integer> filterAllOddNumbers(List<Integer> numbers) {
		return numbers.stream().filter(x -> (x % 2 == 0)).collect(Collectors.toList());
	}

	private static List<Integer> filterAllEvenNumbers(List<Integer> numbers) {
		return numbers.stream().filter(i -> i % 2 != 0).collect(Collectors.toList());
	}

	private static List<Integer> filterAllNumbersLessThanX(List<Integer> numbers, int X) {
		return numbers.stream().filter(i -> i >= X).collect(Collectors.toList());
	}
	
	private static int sumOfAllEvenNumbersSquareInParallel(List<Integer> numbers) {
		return numbers.parallelStream().filter(x -> (x % 2 == 0)).mapToInt(x -> x * x).sum();
	}
	
	private static List<String> convertFromIntToString(List<Integer> values){
		return values.stream().map(String::valueOf).collect(Collectors.toList());
	}

}