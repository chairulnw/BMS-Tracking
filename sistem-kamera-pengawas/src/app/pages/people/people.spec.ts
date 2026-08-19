import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { People } from './people';

describe('People', () => {
  let component: People;
  let fixture: ComponentFixture<People>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [People],
      providers: [provideHttpClient(), provideHttpClientTesting(), provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(People);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('starts with empty search and today as the date range', () => {
    expect(component.searchQuery).toBe('');
    expect(component.fromDate).toBe(component.toDate);
  });
});
