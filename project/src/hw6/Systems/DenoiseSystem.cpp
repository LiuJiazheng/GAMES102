#include "DenoiseSystem.h"

#include "../Components/DenoiseData.h"

#include <_deps/imgui/imgui.h>

#include <spdlog/spdlog.h>

#include <Eigen/Sparse>

#include <utility>

#define VECTOR_UV(U,V) ((V)->position - (U)->position)

constexpr float LAMBDA = 0.005;
constexpr int STEPS = 20;

using namespace Ubpa;
using std::acos;

/* helper function, for color virtualization.*/
rgbf ColorMap(float c) {
	float r = 0.8f, g = 1.f, b = 1.f;
	c = c < 0.f ? 0.f : (c > 1.f ? 1.f : c);

	if (c < 1.f / 8.f) {
		r = 0.f;
		g = 0.f;
		b = b * (0.5f + c / (1.f / 8.f) * 0.5f);
	}
	else if (c < 3.f / 8.f) {
		r = 0.f;
		g = g * (c - 1.f / 8.f) / (3.f / 8.f - 1.f / 8.f);
		b = b;
	}
	else if (c < 5.f / 8.f) {
		r = r * (c - 3.f / 8.f) / (5.f / 8.f - 3.f / 8.f);
		g = g;
		b = b - (c - 3.f / 8.f) / (5.f / 8.f - 3.f / 8.f);
	}
	else if (c < 7.f / 8.f) {
		r = r;
		g = g - (c - 5.f / 8.f) / (7.f / 8.f - 5.f / 8.f);
		b = 0.f;
	}
	else {
		r = r - (c - 7.f / 8.f) / (1.f - 7.f / 8.f) * 0.5f;
		g = 0.f;
		b = 0.f;
	}

	return rgbf{ r,g,b };
}

void MeshToHEMesh(DenoiseData* data) {
	data->heMesh->Clear();

	if (!data->mesh) {
		spdlog::warn("mesh is nullptr");
		return;
	}

	if (data->mesh->GetSubMeshes().size() != 1) {
		spdlog::warn("number of submeshes isn't 1");
		return;
	}

	std::vector<size_t> indices(data->mesh->GetIndices().begin(), data->mesh->GetIndices().end());
	data->heMesh->Init(indices, 3);

	if (!data->heMesh->IsTriMesh())
		spdlog::warn("HEMesh init fail");

	for (size_t i = 0; i < data->mesh->GetPositions().size(); i++) {
		data->heMesh->Vertices().at(i)->position = data->mesh->GetPositions().at(i);
	}
}

void HEMeshToMesh(DenoiseData* data) {
	if (!data->mesh) {
		spdlog::warn("mesh is nullptr");
		return;
	}

	if (!data->heMesh->IsTriMesh() || data->heMesh->IsEmpty()) {
		spdlog::warn("HEMesh isn't triangle mesh or is empty");
		return;
	}

	data->mesh->SetToEditable();

	const size_t N = data->heMesh->Vertices().size();
	const size_t M = data->heMesh->Polygons().size();
	std::vector<Ubpa::pointf3> positions(N);
	std::vector<uint32_t> indices(M * 3);
	for (size_t i = 0; i < N; i++)
		positions[i] = data->heMesh->Vertices().at(i)->position;
	for (size_t i = 0; i < M; i++) {
		auto tri = data->heMesh->Indices(data->heMesh->Polygons().at(i));
		indices[3 * i + 0] = static_cast<uint32_t>(tri[0]);
		indices[3 * i + 1] = static_cast<uint32_t>(tri[1]);
		indices[3 * i + 2] = static_cast<uint32_t>(tri[2]);
	}
	data->mesh->SetPositions(std::move(positions));
	data->mesh->SetIndices(std::move(indices));
	data->mesh->SetSubMeshCount(1);
	data->mesh->SetSubMesh(0, { 0, M * 3 });
	data->mesh->GenNormals();
	data->mesh->GenTangents();
}

float TriangleArea(Vertex* v0, Vertex* v1, Vertex* v2) {
	return 0.5 * v0->position.distance(v1->position) * \
		v0->position.distance(v2->position) * \
		(v1->position - v0->position).sin_theta(v2->position - v0->position);
}

inline Vertex* FindThirdVertex(Vertex* v, Vertex* adj_v) {
	HalfEdge* edge = adj_v->HalfEdge(); // this edge is the one vertex steming to
	while (true) {
		// actually, this loop only has one execution time
		if (v == edge->Next()->End()) {
			return edge->End();
		}
		else {
			edge = edge->Pair()->Next();
		}
	}
}

float MixedVoronoiCellArea(Vertex* v) {
	float area = 0.0f;
	if (v->IsOnBoundary()) return area;
	for (auto* adj_v : v->AdjVertices()) {
		Vertex* third_v = FindThirdVertex(v, adj_v);
		float tri_area = TriangleArea(v, adj_v, third_v);
		if (tri_area < EPSILON<float>) continue; // arae is too small to compute
		if ((adj_v->position - v->position).dot(third_v->position - v->position) > 0.0f &&
			(v->position - adj_v->position).dot(third_v->position - adj_v->position) > 0.0f &&
			(v->position - third_v->position).dot(adj_v->position - third_v->position) > 0.0f) {
			// triangle is non-obtuse 
			area += 1.0F / 8.0F * (
				    adj_v->position.distance2(v->position) * \
				    (adj_v->position - third_v->position).cot_theta(v->position - third_v->position) + \
				    third_v->position.distance2(v->position) * \
				    (v->position - adj_v->position).cot_theta(third_v->position-adj_v->position) );
		}
		else {
			if (VECTOR_UV(v,adj_v).dot(VECTOR_UV(v, third_v)) < 0.0f) {
				// triangle is obtuse and angle at v is obtuse
				area += tri_area / 2.0f;
			}
			else {
				area += tri_area / 4.0f;
			}
		}
	}
	return area;
}

float GaussianCurvature(Vertex* v) {
	if (v->IsOnBoundary()) return 0.0f;
	float area = MixedVoronoiCellArea(v);
	if (area < EPSILON<float>) return 0.0f;
	float angle = 0.0f;
	for (Vertex* adj_v : v->AdjVertices()) {
		Vertex* third_v = FindThirdVertex(v, adj_v);
		// if triangle facet is too small, skip it
		if (TriangleArea(v, adj_v, third_v) < EPSILON<float>) continue;
		angle += acos(VECTOR_UV(v, adj_v).cos_theta(VECTOR_UV(v, third_v)));
	}
	return (2 * PI<float> -angle) / area;
}

inline std::pair<Vertex*, Vertex*> WeightVertexPair(Vertex* v, Vertex* adj_v) {
	HalfEdge* edge = adj_v->HalfEdge();
	Vertex* alpha_v = nullptr;
	Vertex* beta_v = nullptr;
	while (true) {
		if (v == edge->End()) {
			beta_v = edge->Next()->End();
			break;
		}
		else {
			edge = edge->Pair()->Next();
		}
	}
	while (true) {
		if (v == edge->Next()->End()) {
			alpha_v = edge->End();
			break;
		}
		else {
			edge = edge->Pair()->Next();
		}
	}
	return std::make_pair(alpha_v, beta_v);
}

valf3 MeanCurvatureOperator(Vertex* v) {
	valf3 c{ 0.0f };
	if (v->IsOnBoundary()) return c;
	float area = MixedVoronoiCellArea(v);
	if (area < EPSILON<float>) return c;
	for (Vertex* adj_v : v->AdjVertices()) {
		// we need the vertex which is opposite to third_v
		auto temp = WeightVertexPair(v, adj_v);
		Vertex* alpha_v = temp.first;
		Vertex* beta_v = temp.second;
		// if triangle facet is too small, skip it
		if (TriangleArea(v, adj_v, beta_v) < EPSILON<float> || 
			TriangleArea(v, adj_v, alpha_v) < EPSILON<float>) continue;
		// now we have alpha and beta vertices, cacluate one piece
		float cot_alpha = VECTOR_UV(alpha_v, adj_v).cot_theta(VECTOR_UV(alpha_v, v));
		float cot_beta = VECTOR_UV(beta_v, adj_v).cot_theta(VECTOR_UV(beta_v, v));
		c += (cot_alpha + cot_beta) * VECTOR_UV(adj_v,v);
	}
	return 1.0f / (2.0f * area) * c;
}

inline void MimimalSurfaceIterating(Vertex* v, float lambda) {
	v->position = v->position + (-1.0f) * lambda * MeanCurvatureOperator(v);
}

#define MeshIndex(v) static_cast<int>(mesh->Index(v))

void GlobalMiminmalSurfaceSmoothing(DenoiseData* data) {
	// shared pointer of he mesh
	auto mesh = data->heMesh;
	size_t n = mesh->Vertices().size();
	Eigen::MatrixX3f B(n, 3);
	std::vector<Eigen::Triplet<float> > A_triplet;
	for (auto* v : mesh->Vertices()) {
		if (v->IsOnBoundary()) {
			A_triplet.push_back({MeshIndex(v), MeshIndex(v), 1.0 });
			B.row(MeshIndex(v)) = Eigen::Vector3f(v->position[0],v->position[1],v->position[2]);
		}
		else {
			float w = 0.0f;
			for (auto* adj_v : v->AdjVertices()) {
				// we need the vertex which is opposite to third_v
				auto temp = WeightVertexPair(v, adj_v);
				Vertex* alpha_v = temp.first;
				Vertex* beta_v = temp.second;
				// now we have alpha and beta vertices, cacluate one piece
				float cot_alpha = VECTOR_UV(alpha_v, adj_v).cot_theta(VECTOR_UV(alpha_v, v));
				float cot_beta = VECTOR_UV(beta_v, adj_v).cot_theta(VECTOR_UV(beta_v, v));
				A_triplet.push_back({ MeshIndex(v) , MeshIndex(adj_v), cot_alpha + cot_beta });
				w += cot_alpha + cot_beta;
				mesh->Index(v);
			}
			A_triplet.push_back({ MeshIndex(v), MeshIndex(v), -w });
			B.row(MeshIndex(v)) = Eigen::Vector3f::Zero();
		}
	}
	
	Eigen::SparseMatrix<float> A(n, n);
	A.setFromTriplets(A_triplet.begin(),A_triplet.end());
	Eigen::SparseLU<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int> >solver;
	solver.analyzePattern(A);
	solver.factorize(A);
	Eigen::MatrixXf X = solver.solve(B);
	
	if (solver.info() != Eigen::Success)
	{
		spdlog::warn("SurfaceSmoothing: Could not solve linear system\n");
	}
	else
	{
		// copy solution
		for (auto* v : mesh->Vertices()) {
			// vec must be a Vector3f
			auto vec = X.row(MeshIndex(v));
			v->position = Ubpa::pointf3{vec[0], vec[1], vec[2]};
		}
	}
}

void DenoiseSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<DenoiseData>();
		if (!data)
			return;

		if (ImGui::Begin("Denoise")) {
			if (ImGui::Button("Mesh to HEMesh")) {
				data->heMesh->Clear();
				[&]() {
					MeshToHEMesh(data);

					spdlog::info("Mesh to HEMesh success");
				}();
			}

			if (ImGui::Button("Add Noise")) {
				[&]() {
					if (!data->heMesh->IsTriMesh()) {
						spdlog::warn("HEMesh isn't triangle mesh");
						return;
					}

					for (auto* v : data->heMesh->Vertices()) {
						v->position += data->randomScale * (
							2.f * Ubpa::vecf3{ Ubpa::rand01<float>(),Ubpa::rand01<float>() ,Ubpa::rand01<float>() } - Ubpa::vecf3{ 1.f }
						);
					}

					spdlog::info("Add noise success");
				}();
			}

			if (ImGui::Button("Set Normal to Color")) {
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}

					data->mesh->SetToEditable();
					const auto& normals = data->mesh->GetNormals();
					std::vector<rgbf> colors;
					for (const auto& n : normals)
						colors.push_back((n.as<valf3>() + valf3{ 1.f }) / 2.f);
					data->mesh->SetColors(std::move(colors));

					spdlog::info("Set Normal to Color Success");
				}();
			}

			if (ImGui::Button("Gaussian Curvature Coloring")) {
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}
					data->mesh->SetToEditable();
					MeshToHEMesh(data);
					if (!data->heMesh->IsTriMesh() || data->heMesh->IsEmpty()) {
						spdlog::warn("HEMesh isn't triangle mesh or is empty");
						return;
					}
					std::vector<rgbf> color_map;
					for (Vertex* v : data->heMesh->Vertices()) {
						color_map.push_back(ColorMap(GaussianCurvature(v)));
					}
					data->mesh->SetColors(std::move(color_map));
					HEMeshToMesh(data);

					spdlog::info("Set Gaussian Curvature to Color Success");
				}();
			}

			if (ImGui::Button("Mean Curvature Coloring")) {
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}
					data->mesh->SetToEditable();
					MeshToHEMesh(data);
					if (!data->heMesh->IsTriMesh() || data->heMesh->IsEmpty()) {
						spdlog::warn("HEMesh isn't triangle mesh or is empty");
						return;
					}
					std::vector<rgbf> color_map;
					for (Vertex* v : data->heMesh->Vertices()) {
						color_map.push_back(ColorMap(MeanCurvatureOperator(v).norm()/2.0f));
					}
					data->mesh->SetColors(std::move(color_map));
					HEMeshToMesh(data);

					spdlog::info("Set Mean Curvature to Color Success");
				}();
			}

			if (ImGui::Button("Minimal Surface")) {
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}
					// back up
					data->copy = *data->mesh;
					MeshToHEMesh(data);
					if (!data->heMesh->IsTriMesh() || data->heMesh->IsEmpty()) {
						spdlog::warn("HEMesh isn't triangle mesh or is empty");
						return;
					}
					
					for (int i = 0; i < data->num_iterations; i++) {
						for (Vertex* v : data->heMesh->Vertices()) {
							if(!v->IsOnBoundary())
								MimimalSurfaceIterating(v, data->lambda);
						}
					}
					HEMeshToMesh(data);
					spdlog::info("Set to Minimal Surface Success");
				}();
			}

			if (ImGui::Button("Global Miminal Surface")) {
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}
					// back up
					data->copy = *data->mesh;
					MeshToHEMesh(data);
					if (!data->heMesh->IsTriMesh() || data->heMesh->IsEmpty()) {
						spdlog::warn("HEMesh isn't triangle mesh or is empty");
						return;
					}
					GlobalMiminmalSurfaceSmoothing(data);
					HEMeshToMesh(data);
					spdlog::info("Set to Global Minimal Surface Success");
				}();
			}

			if (ImGui::Button("HEMesh to Mesh")) {
				[&]() {
					HEMeshToMesh(data);

					spdlog::info("HEMesh to Mesh success");
				}();
			}

			if (ImGui::Button("Recover Mesh")) {
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}
					if (data->copy.GetPositions().empty()) {
						spdlog::warn("copied mesh is empty");
						return;
					}

					*data->mesh = data->copy;

					spdlog::info("recover success");
				}();
			}
		}
		ImGui::End();
	});
}

