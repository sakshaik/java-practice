package com.pattern.factory;

public class SportNotFound implements Sport {

	@Override
	public void substitutes() {
		System.out.println("Sport not supported");
	}

}
