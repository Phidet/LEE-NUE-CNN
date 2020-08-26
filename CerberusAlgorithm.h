/**
 *  @file   larpandoracontent/LArWorkshop/CerberusAlgorithm.h
 *
 *  @brief  Header file for the Cerberus algorithm class.
 *
 *  $Log: $ 
 */

#ifndef LAR_CERBERUS_ALGORITHM_H
#define LAR_CERBERUS_ALGORITHM_H 1
#include "Pandora/Algorithm.h"
#include <torch/script.h>
#include <torch/torch.h> // Only used for testing (saving tensors to file with torch::save())

#define IMSIZE 384 //256*1.5 Size of the generated image arrays
#define SEG 128

namespace lar_content
{
/** 
 *  @brief  CerberusAlgorithm class 
 */

	class CerberusAlgorithm : public pandora::Algorithm
	{
	public:
/**     
 *  @brief  Factory class for instantiating algorithm     
 */

		class Factory : public pandora::AlgorithmFactory
		{
		public:
			pandora::Algorithm *CreateAlgorithm() const;
		};

	private:
		pandora::StatusCode Run();
		pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);
		pandora::StatusCode WriteDetectorGaps(torch::Tensor &tensor, const float minZ_U, const float minZ_V, const float minZ_W);
		pandora::StatusCode PopulateImage(torch::Tensor &tensor, const pandora::CaloHitVector &caloHitVector, const int index, const float minX, const float minZ);
		pandora::StatusCode PopulateAvailabilityTensor(torch::Tensor &tensor, const pandora::CaloHitVector &caloHitVector, const int index, const float minX, const float minZ);
		pandora::StatusCode PopulatePandoraReconstructionTensor(torch::Tensor &tensor, const pandora::PfoList *const pPfoList, const pandora::HitType tpcView, const int index, const float minX, const float minZ);
		pandora::StatusCode PopulateMCTensor(torch::Tensor &tensor, const pandora::CaloHitVector &caloHitVector, const int index, const float minX, const float minZ);
		pandora::StatusCode Backtracing(const torch::Tensor &tensor, pandora::CaloHitList &caloHitListChange, const pandora::PfoList *pPfoList, const float minX, const float minZ, const pandora::HitType tpcView, const pandora::CartesianVector ShowerVertex2D);
		pandora::StatusCode FindSuitableCluster(const pandora::CaloHit *const pCaloHit, const pandora::Cluster *&pBestCluster, const int caloHitClass, const float maxDistance);
		pandora::StatusCode CaloHitReallocation(const torch::Tensor &tensor, const pandora::ClusterList *const pClusterListTemp, const pandora::HitType tpcView, const float minX, const float minZ);
		pandora::StatusCode ClusterCreation(const pandora::CaloHit *const pCaloHit, const int caloHitClass);
		pandora::StatusCode PfoCreation(const pandora::ParticleFlowObject *pNeutrinoPfo);


		void MatchingShowerReconstructionPercentage(const torch::Tensor &tensor, const pandora::PfoList *pPfoList, const pandora::HitType tpcView, const float minX, const float minZ, float &showerMatchValue);
		bool inViewXZ(int &x, int &z, const pandora::CaloHit *const pCaloHit, const float minX, const float minZ);
		void FillMinimizationArray(std::array<float, 128> &hitDensity, const pandora::PfoList *const pPfoList, const pandora::CaloHitList *const pCaloHitList, const pandora::CartesianVector v, const float startD1, const float startD2, const bool directionX, const pandora::HitType tpcView);
		float FindMin(const std::array<float, 128> hitDensity, const float startPoint) const;
		// Member variables here
		std::string m_pfoListName;
		std::string m_outputClusterListName;
		std::string m_outputPfoListName;
		pandora::StringVector m_caloHitListNames;
		//pandora::StringVector m_clusterListNames;
	};
//------------------------------------------------------------------------------------------------------------------------------------------
	inline pandora::Algorithm *CerberusAlgorithm::Factory::CreateAlgorithm() const
	{
		return new CerberusAlgorithm();
	}

} // namespace lar_content
#endif // #ifndef LAR_CERBERUS_ALGORITHM_H
