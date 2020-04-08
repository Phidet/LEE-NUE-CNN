/** 
*  @file   larpandoracontent/LArWorkshop/TrainingExportAlgorithm.cc 
* 
*  @brief  Implementation of the TrainingExport algorithm class. 
* 
*  $Log: $ */
#include "Pandora/AlgorithmHeaders.h"
#include "larpandoracontent/MyArea/TrainingExportAlgorithm.h"
#include "PandoraMonitoringApi.h"
#include "larpandoracontent/LArHelpers/LArPfoHelper.h"
#include "larpandoracontent/LArHelpers/LArClusterHelper.h"
#include "larpandoracontent/LArHelpers/LArInteractionTypeHelper.h"
#include "larpandoracontent/LArHelpers/LArMonitoringHelper.h"
#include "Objects/MCParticle.h"
#include <fstream>
#include <array>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace pandora;

namespace lar_content
{
	StatusCode TrainingExportAlgorithm::Run()
	{		
		const CaloHitList *pCaloHitListU(nullptr);
		const CaloHitList *pCaloHitListV(nullptr);
		const CaloHitList *pCaloHitListW(nullptr);
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNameU, pCaloHitListU));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNameV, pCaloHitListV));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNameW, pCaloHitListW));

		CaloHitVector caloHitVectorU(pCaloHitListU->begin(), pCaloHitListU->end());
		CaloHitVector caloHitVectorV(pCaloHitListV->begin(), pCaloHitListV->end());
		CaloHitVector caloHitVectorW(pCaloHitListW->begin(), pCaloHitListW->end());

		std::sort(caloHitVectorU.begin(), caloHitVectorU.end(), LArClusterHelper::SortHitsByPosition);

		std::array<std::array<float,IMSIZE>,IMSIZE> viewU = {0};
		//float viewU[IMSIZE][IMSIZE] = {0};
		float labelU[IMSIZE][IMSIZE][3] = {0};

		PopulateImage(caloHitVectorU, viewU, labelU);
		//cv::Mat M(512, 512, CV_32FC1, black);
		//cv::imwrite("~/imgOut.bmp",  M);
		std::ofstream file("TrainingTestingOutput/viewU.bin", std::ios::out | std::ios::binary);  // https://stackoverflow.com/questions/48193667/save-array-to-binary-file-not-working-c
		if(!file) {
    		// error handling
			std::cout << "ERROR" <<std::endl;
		}
		//file.write(viewU[0][0], IMSIZE * IMSIZE * sizeof(decltype(viewU)::value_type));

		return STATUS_CODE_SUCCESS;
	}

StatusCode TrainingExportAlgorithm::MinBoundaries(const CaloHitVector &caloHitVector, float &minX, float &minZ)
{
	/*for (const CaloHit *const pCaloHit : caloHitVector)
	{
		pCaloHit
	}*/
	minX = caloHitVector.front()->GetPositionVector().GetX();
	minZ = caloHitVector.front()->GetPositionVector().GetZ();
	return STATUS_CODE_SUCCESS;
}


StatusCode TrainingExportAlgorithm::PopulateImage(const CaloHitVector &caloHitVector, std::array<std::array<float,IMSIZE>,IMSIZE> &view, float (&label)[IMSIZE][IMSIZE][3])
{
	float minX, minZ;
	(void) MinBoundaries(caloHitVector, minX, minZ);
	for (const CaloHit *const pCaloHit : caloHitVector)
	{
    	const int i = (pCaloHit->GetPositionVector().GetX()-minX)/0.03; // Pixel number in X direction
    	const int j = (pCaloHit->GetPositionVector().GetZ()-minZ)/0.03; // Pixel number in Y direction
		const MCParticleWeightMap  &mcParticleWeightMap(pCaloHit->GetMCParticleWeightMap());

		// Populates input image
    	view[i][j] = pCaloHit->GetHadronicEnergy();
    	
    	// Populates prediction image
    	for (const MCParticleWeightMap::value_type &mapEntry : mcParticleWeightMap)
    	{
    		const int particleID = mapEntry.first->GetParticleId();
    		switch(particleID){
				case 22: case 11: case -11:
					label[i][j][0] += mapEntry.second;
					break;
				case 2212:
					label[i][j][1] += mapEntry.second;
					break;
				default:
					label[i][j][2] += mapEntry.second;
    		}
    	}
	}
	return STATUS_CODE_SUCCESS;
}
//------------------------------------------------------------------------------------------------------------------------------------------
	StatusCode TrainingExportAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
	{
// Read settings from xml file here
//PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "PfoListName", m_pfoListName));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "CaloHitListNameU", m_caloHitListNameU));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "CaloHitListNameV", m_caloHitListNameV));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "CaloHitListNameW", m_caloHitListNameW));

		return EventValidationAlgorithm::ReadSettings(xmlHandle);
	}
} // namespace lar_content
