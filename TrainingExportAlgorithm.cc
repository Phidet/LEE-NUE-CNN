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

//#include "larpandora/LArPandoraInterface/LArPandoraGeometry.h"

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

		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateImage(caloHitVectorU));

		return STATUS_CODE_SUCCESS;
	}


StatusCode TrainingExportAlgorithm::PopulateImage(const CaloHitVector &caloHitVector)
{

	 // // Get global TPC geometry information
  //   const LArTPCMap &larTPCMap(this->GetPandora().GetGeometry()->GetLArTPCMap());
  //   const LArTPC *const pFirstLArTPC(larTPCMap.begin()->second);

  //   const float minX(pFirstLArTPC->GetCenterX() - 0.5f * pFirstLArTPC->GetWidthX());
  //   const float widthX(pFirstLArTPC->GetWidthX());
  //   //const float minY(pFirstLArTPC->GetCenterY() - 0.5f * pFirstLArTPC->GetWidthY());
  //   //const float widthY(pFirstLArTPC->GetWidthY());
  //   const float minZ(pFirstLArTPC->GetCenterZ() - 0.5f * pFirstLArTPC->GetWidthZ());
  //   const float widthZ(pFirstLArTPC->GetWidthZ());

	std::ofstream file("OutTest/viewU.bin", std::ios::out | std::ios::binary | std::ios::app); 
	if(!file)
	{
		std::cout<<"Problem opening/creating binary file in TrainingExportAlgorithm::PopulateImage."<<std::endl;
		return STATUS_CODE_FAILURE;
	}

	const float hitNumber = 1.22f;//caloHitVector.size();
	file.write((char*)&hitNumber, sizeof(hitNumber));

	for (const CaloHit *const pCaloHit : caloHitVector)
	{
		std::array<float, 4> pixel = {0};

    	const float x = pCaloHit->GetPositionVector().GetX(); //(pCaloHit->GetPositionVector().GetX()-minX)/widthX; // Pixel number in X direction
    	const float z = pCaloHit->GetPositionVector().GetZ(); //(pCaloHit->GetPositionVector().GetZ()-minZ)/widthZ; // Pixel number in Z direction
    	file.write((char*)&x, sizeof(x));
    	file.write((char*)&z, sizeof(z));
		
		const MCParticleWeightMap  &mcParticleWeightMap(pCaloHit->GetMCParticleWeightMap());

		// Populates input image
    	pixel[0] = pCaloHit->GetHadronicEnergy();

    	// Populates prediction image
    	for (const MCParticleWeightMap::value_type &mapEntry : mcParticleWeightMap)
    	{
    		const int particleID = mapEntry.first->GetParticleId();
    		switch(particleID){
				case 22: case 11: case -11:
					pixel[1] += mapEntry.second;
					break;
				case 2212:
					pixel[2] += mapEntry.second;
					break;
    		}
    	}
    	pixel[3] = 1.0f - pixel[1] - pixel[2];
    	file.write((char*)&pixel, sizeof(pixel));
	}
	file.close();
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

		return STATUS_CODE_SUCCESS;
	}
} // namespace lar_content
