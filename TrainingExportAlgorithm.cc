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
#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "Objects/MCParticle.h"
#include <fstream>
#include <array>
#include <limits>

#ifdef MONITORING
#include "PandoraMonitoringApi.h"
#endif

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

		const PfoList *pPfoList(nullptr);
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pPfoList));
		
		float minX(0);
		float minZ(0);
		OneShowerMinBoundaries(pPfoList, minX, minZ);
		// float minX, minZ;
		// if(OneShowerMinBoundaries(pPfoList, minX, minZ))
		// {
		// }
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateImage(caloHitVectorU, minX, minZ));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateRecoImage(*pPfoList));

		// PANDORA_MONITORING_API(VisualizeCaloHits(this->GetPandora(), pCaloHitListU, std::string("ViewU"), BLUE));
		// const VertexList *pVertexList(nullptr);
		// PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pVertexList));
		// PANDORA_MONITORING_API(VisualizeVertices(this->GetPandora(), pVertexList, std::string("VertexList"), GREEN));
		// const MCParticleList *pMCParticleList(nullptr);
		// PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pMCParticleList));
		// PANDORA_MONITORING_API(VisualizeMCParticles(this->GetPandora(), pMCParticleList, std::string("MCParticleList"), YELLOW));

        // CartesianVector dl = CartesianVector(minX, 0, minZ);
        // CartesianVector dr = CartesianVector(minX+78, 0, minZ);
        // CartesianVector ul = CartesianVector(minX, 0, minZ+78);
        // CartesianVector ur = CartesianVector(minX+78, 0, minZ+78);

	 	// PANDORA_MONITORING_API(AddLineToVisualization(this->GetPandora(), &dl, &ul, std::string("left"), GRAY,1,1));
	 	// PANDORA_MONITORING_API(AddLineToVisualization(this->GetPandora(), &dr, &ur, std::string("right"), GRAY,1,1));
	 	// PANDORA_MONITORING_API(AddLineToVisualization(this->GetPandora(), &dl, &dr, std::string("bottom"), GRAY,1,1));
	 	// PANDORA_MONITORING_API(AddLineToVisualization(this->GetPandora(), &ul, &ur, std::string("top"), GRAY,1,1));
	 	// PANDORA_MONITORING_API(ViewEvent(this->GetPandora()));

		return STATUS_CODE_SUCCESS;
	}

	bool TrainingExportAlgorithm::OneShowerMinBoundaries(const PfoList *const pPfoList, float &minX, float &minZ)
	{
		PfoList pfoListCrop;
		//const ParticleFlowObject *pShowerPfo(nullptr);
		float vertexX(std::numeric_limits<float>::max());
		float vertexZ(std::numeric_limits<float>::max());
		for (const ParticleFlowObject *const pPfo : *pPfoList) // Finds and adds shower to pfoListCrop
		{
			if(LArPfoHelper::IsNeutrinoFinalState(pPfo))
			{
				pfoListCrop.push_back(pPfo);

				if (LArPfoHelper::IsShower(pPfo)) // && LArPfoHelper::IsNeutrinoFinalState(pPfo)
				{	    		
					CartesianVector v =  LArPfoHelper::GetVertex(pPfo)->GetPosition();
					v = LArGeometryHelper::ProjectPosition(this->GetPandora(), v, TPC_VIEW_U); // Project 3D vertex onto 2D U view
					if(v.GetZ()<vertexZ)
					{
						vertexX = v.GetX();
						vertexZ = v.GetZ();
					}
				}
			}
		}

		if(vertexZ==std::numeric_limits<float>::max()) // If no shower is found
		{
			return false;
		}

		std::cout<<"------------------ Point T1.11"<<std::endl;
		CaloHitList caloHitListInCrop;
		std::cout<<"------------------ Point T1.11.1"<<std::endl;
		LArPfoHelper::GetCaloHits(pfoListCrop, TPC_VIEW_U, caloHitListInCrop);
		std::cout<<"------------------ Point T1.12"<<std::endl;
		
		const int seg = 128;
	    std::array<uint, seg>  hitXDensity= {0}; // Always combining 8 wires
	    for (const CaloHit *const pCaloHit : caloHitListInCrop)
	    {
	    	const float x = pCaloHit->GetPositionVector().GetX();
	    	const float z = pCaloHit->GetPositionVector().GetZ();
	    	const int pixelX = (int) ((x-vertexX)/0.3 + IMSIZE)/(2*IMSIZE/seg);
	    	if(pixelX>=0 && pixelX<seg && (z-minZ)/0.3<IMSIZE && (z-minZ)/0.3>=0)
	    		hitXDensity[pixelX]++;
	    }
	    std::cout<<"------------------ Point T1.14"<<std::endl;
	    int left(0);
	    int right(seg/2);
	    int loopCounter(1);
	    int leftTotal, rightTotal;
	    do{
	    	leftTotal = 0;
	    	rightTotal = 0;
	    	for(int i=0; i<seg/2; i++)
	    	{
	    		leftTotal += hitXDensity[left+i];
	    		rightTotal += hitXDensity[right+i];
	    	}

	    	if(leftTotal>rightTotal) right -= (seg/4)/loopCounter;
	    	else left += (seg/4)/loopCounter;
	    	loopCounter *=2;
	    	std::cout<<"^^^^^^^^^^^ ResultLeft: "<<left <<" "<<vertexX<<" "<<seg<<std::endl;
	    } while(loopCounter<=(seg/2));
	    std::cout<<"------------------ Point T1.15"<<std::endl;
	    minX = ((2.0*left)/seg-1) * IMSIZE * 0.3 + vertexX;
		minZ = vertexZ-5.0; // takes vertex z position as shower start
		return true;
	}


	StatusCode TrainingExportAlgorithm::PopulateImage(const CaloHitVector &caloHitVector, const float minX, const float minZ)
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

	file.write((char*)&minX, sizeof(minX));
	file.write((char*)&minZ, sizeof(minZ));

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
    			case 22: case 11: case -11: case 211: case -211:
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


StatusCode TrainingExportAlgorithm::PopulateRecoImage(const PfoList &pfoList)
{

	std::ofstream file("OutTest/PandoraRecoU.bin", std::ios::out | std::ios::binary | std::ios::app); 
	if(!file)
	{
		std::cout<<"Problem opening/creating binary file in TrainingExportAlgorithm::PopulateImage."<<std::endl;
		return STATUS_CODE_FAILURE;
	}

	const float hitNumber = 1.22f;
	file.write((char*)&hitNumber, sizeof(hitNumber));

	for (const ParticleFlowObject *const pPfo: pfoList)
	{
		PfoList pfoListTemp;
		pfoListTemp.push_back(pPfo);
		CaloHitList caloHitList;
		LArPfoHelper::GetCaloHits(pfoListTemp, TPC_VIEW_U, caloHitList);
		for (const CaloHit *const pCaloHit : caloHitList)
		{
			std::array<float, 4> pixel = {0};
    		const float x = pCaloHit->GetPositionVector().GetX(); //(pCaloHit->GetPositionVector().GetX()-minX)/widthX; // Pixel number in X direction
    		const float z = pCaloHit->GetPositionVector().GetZ(); //(pCaloHit->GetPositionVector().GetZ()-minZ)/widthZ; // Pixel number in Z direction
    		file.write((char*)&x, sizeof(x));
    		file.write((char*)&z, sizeof(z));

			if(LArPfoHelper::IsShower(pPfo))
			{
				if(LArPfoHelper::IsNeutrinoFinalState(pPfo))
				{
					pixel[0] = 1.0f;
				}
				else
				{
					pixel[1] = 1.0f;	
				}
			}
			else if(LArPfoHelper::IsTrack(pPfo))
			{
				if(LArPfoHelper::IsNeutrinoFinalState(pPfo))
				{
					pixel[2] = 1.0f;
				}
				else
				{
					pixel[3] = 1.0f;	
				}
			}
    		file.write((char*)&pixel, sizeof(pixel));
    	}
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
