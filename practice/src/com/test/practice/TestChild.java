package com.test.practice;

public class TestChild extends TestParent {

	public TestChild(String test) {
		super(test);
		System.out.println("This is child Constructor " + test);
		super.printMessage();
	}

	public static void main(String[] args) {
		TestParent parent = new TestChild("new");
		parent.printMessage();
	}

	public void printMessage() {
		System.out.println("This is child");
		super.printMessage();
	}

}