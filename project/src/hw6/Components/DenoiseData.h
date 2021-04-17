#pragma once

#include <Utopia/Render/Mesh.h>
#include <Utopia/App/Editor/InspectorRegistry.h>
#include "../HEMeshX.h"

struct DenoiseData {
	// [[...]] is attribute list.
	// User-defined attributes are useless for the compiler,
	// but Utopia use these informations for Inspector.
	// If you change struct DenoiseData, you should delete 'details/DenoiseData_AutoRefl.inl'
	// or right click 'DenoiseData_AutoRefl.inl.rule' in 'solution explorer' and select 'compile'
	// CMake will generate a new one.

	// [[UInspector::range(std::pair{0.f, 10.f})]]
	[[UInspector::min_value(0.f)]]
	[[UInspector::tooltip("random scale")]]
	float randomScale = 1.f;

	std::shared_ptr<Ubpa::Utopia::Mesh> mesh;

	[[UInspector::hide]]
	std::shared_ptr<HEMeshX> heMesh{ std::make_shared<HEMeshX>() };

	[[UInspector::hide]]
	Ubpa::Utopia::Mesh copy;

	//Params for HW6
	[[UInspector::min_value(0)]]
	[[UInspector::step(1)]]
	[[UInspector::tooltip("number of iterations")]]
	int num_iterations{ 1 };

	[[UInspector::min_value(0)]]
	[[UInspector::step(0.01)]]
	[[UInspector::tooltip("lambda")]]
	float lambda{ 0.005f };

	[[UInspector::min_value(0)]]
	[[UInspector::max_value(2)]]
	[[UInspector::step(1)]]
	[[UInspector::tooltip("harmonic mapping mode")]]
	int harmonic_mode{ 1 };
};

#include "details/DenoiseData_AutoRefl.inl"
