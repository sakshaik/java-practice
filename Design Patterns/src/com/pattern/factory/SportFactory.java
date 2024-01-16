package com.pattern.factory;

public class SportFactory {

	private static SportFactory instance = null;

	private SportFactory() {
	};

	public static SportFactory getInstance() {
		if (instance == null) {
			instance = new SportFactory();
		}
		return instance;
	}

	public Sport getSport(String sport) {
		if (sport == null) {
			return new SportNotFound();
		}

		if (sport.equalsIgnoreCase("Football")) {
			return new Football();
		}

		if (sport.equalsIgnoreCase("Cricket")) {
			return new Cricket();
		}

		if (sport.equalsIgnoreCase("Rugby")) {
			return new Rugby();
		}
		
		return new SportNotFound();
	}
}
