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
#include <cmath>

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


	bool OneShowerMinBoundaries(float &minX, float &minZ)
	{
		//const pfoList *pPfoListInCrop(nullptr);
		const pfoList *pPfoList(nullptr);
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pPfoList));

		const ParticleFlowObject *pShowerPfo(nullptr);
		for (const ParticleFlowObject *const pPfo : *pPfoList)
		{
			if (LArPfoHelper::IsShower(pPfo)){
	    		if(!pShowerPfo) // If more than one shower is found
	    			return false;
	    		pShowerPfo = pPfo;
	    		//pPfoListInCrop->push_back(pPfo);
	    	}
	    }

	    const CaloHitList *pCaloHitListInCrop(nullptr);
	    LArPfoHelper::GetCaloHits(pShowerPfo, TPC_VIEW_U, pCaloHitListInCrop);
	    
	    const CartesianVector v =  LArPfoHelper::GetVertex(pPfo).GetPosition();
	    const float vertexX = v.GetX();
	    minZ = v.GetZ(); // takes vertex z position as shower start
	    std::array<uint, 128>  hitXDensity= {0}; // Always combining 8 wires

	    for (const CaloHit *const pCaloHit : *pCaloHitListInCrop)
	    {
	    	float x = pCaloHit->GetPositionVector().Get(X)
	    	const int pixelX = (int) (x-vertexX)/(0.3*8) + 0.5; // The 0.5 ensures correct rounding
	    	if(pixelX>=0 && pixelX<=128)
	    		hitXDensity[pixelX]++;
	    }

	    uint left(0);
	    uint right(512);
	    uint loopCounter(1);
	    do{
	    	uint leftTotal(0);
	    	uint rightTotal(0);
	    	for(int i=0; i<128; i++){
	    		leftTotal += hitXDensity[left+i];
	    		rightTotal += hitXDensity[right+i];
	    	}

	    	if(leftTotal>rightTotal) right -= 64/loopCounter;
	    	else left += 64/loopCounter;
	    	loopCounter *=2;
	    } while(leftTotal-rightTotal>10 && loopCounter<=64)

	    minX = left * (0.3*8) + vertexX;
	    return true;
	    // for (const ParticleFlowObject *const pPfo : *pPfoList)
	    // {
	    // 	if (LArPfoHelper::IsNeutrinoFinalState(pPfo) && GetThreeDSeparation(pPfo, pPfoListInCrop->first())<30){
	    // 		pPfoListInCrop->push_back(pPfo);
	    // 	}
	    // }

	    // const CaloHitList *pCaloHitListInCrop(nullptr);
	    // LArPfoHelper::GetCaloHits(pPfo, TPC_VIEW_U, pCaloHitListInCrop);
	    // for (const CaloHit *const pCaloHit : *pPfoList)
	    // {
	    // 	if (LArPfoHelper::IsNeutrinoFinalState(pPfo) && GetThreeDSeparation(pPfo, pPfoListInCrop->first())<30){
	    // 		pPfoListInCrop->push_back(pPfo);
	    // 	}
	    // }
	    // 		inCropPfos.push_back(pPfo);
	    // 		LArPfoHelper::GetVertex(pPfo);
	    // 		LArPfoHelper::IsNeutrinoFinalState(pPfo);
	    // 	else if()
	    // 		inCropPfos.push_back(pPfo);
	    // }
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
